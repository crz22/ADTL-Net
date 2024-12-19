import os
import math

import numpy as np
import torch
import torch.nn.functional as F



def cut_image(img, block_size=(32, 64, 64), step=(32, 64, 64), pad_model='reflect'):
    z_size, x_size, y_size = block_size
    z_step, x_step, y_step = step
    z_img, x_img, y_img = img.shape[2:5]

    z_max = math.ceil((z_img - z_size) / z_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1

    max_num = [z_max, x_max, y_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    x_pad = (x_max - 1) * x_step + x_size - x_img
    y_pad = (y_max - 1) * y_step + y_size - y_img

    if pad_model == 'constant':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')

    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img[:, :, zz * z_step:zz * z_step + z_size, xx * x_step:xx * x_step + x_size,
                            yy * y_step:yy * y_step + y_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image(img_block, block_num, max_num, image_size=(100, 1024, 1024), step=(32, 64, 64)):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step

    zz = 0
    yy = 0
    xx = 0

    for i in range(block_num):
        img_block[i] = img_block[i][:, :, 0:z_step, 0:x_step, 0:y_step]
        if zz == 0:
            image_z = img_block[i]
        else:
            image_z = torch.cat([image_z, img_block[i]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            if yy == 0:
                image_y = image_z
            else:
                image_y = torch.cat([image_y, image_z], dim=4)
            yy = yy + 1

        if yy == y_num:
            # print(image_x)
            yy = 0
            if xx == 0:
                image_x = image_y
            else:
                image_x = torch.cat([image_x, image_y], dim=3)
            xx = xx + 1

    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    # print(image.shape)
    return image

def cut_image_overlap(img, block_size, step, pad_model='reflect'):
    z_size, x_size, y_size = block_size
    z_step, x_step, y_step = step
    z_img, x_img, y_img = img.shape[2:5]

    z_max = math.ceil((z_img - z_size) / z_step) + 1
    x_max = math.ceil((x_img - x_size) / x_step) + 1
    y_max = math.ceil((y_img - y_size) / y_step) + 1

    max_num = [z_max, x_max, y_max]

    z_pad = (z_max - 1) * z_step + z_size - z_img
    x_pad = (x_max - 1) * x_step + x_size - x_img
    y_pad = (y_max - 1) * y_step + y_size - y_img

    if pad_model == 'constant':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'constant', value=-1)
    elif pad_model == 'reflect':
        img = F.pad(img, (0, x_pad, 0, y_pad, 0, z_pad), 'reflect')
    #print(img.shape)
    image_blockes = []
    block_num = 0
    for xx in range(x_max):
        for yy in range(y_max):
            for zz in range(z_max):
                img_block = img[:, :, zz * z_step:zz * z_step + z_size, xx * x_step:xx * x_step + x_size,
                            yy * y_step:yy * y_step + y_size]
                image_blockes.append(img_block)
                block_num = block_num + 1

    return image_blockes, block_num, max_num

def splice_image_overlap(img_block, block_num, max_num, image_size, step):
    z_img, x_img, y_img = image_size
    z_num, x_num, y_num = max_num
    z_step, x_step, y_step = step
    z_block_size, x_block_size, y_block_size = img_block[0].shape[2:5]
    thz = z_block_size - z_step
    thx = x_block_size - x_step
    thy = y_block_size - y_step

    zz = 0
    yy = 0
    xx = 0

    for i in range(block_num):
        if zz == 0:
            image_z = img_block[i]
        else:
            cur_z = image_z.shape[2]
            image_z[:, :, -thz:cur_z, :, :] = (image_z[:, :, -thz:cur_z, :, :] + img_block[i][:, :, 0:thz, :, :]) / 2
            image_z = torch.cat([image_z, img_block[i][:, :, thz:z_block_size, :, :]], dim=2)
        zz = zz + 1
        if zz == z_num:
            zz = 0
            if yy == 0:
                image_y = image_z
            else:
                cur_y = image_y.shape[4]
                image_y[:, :, :, :, -thy:cur_y] = (image_y[:, :, :, :, -thy:cur_y] + image_z[:, :, :, :, 0:thy]) / 2
                image_y = torch.cat([image_y, image_z[:, :, :, :, thy:y_block_size]], dim=4)
            yy = yy + 1

        if yy == y_num:
            # print(image_x)
            yy = 0
            if xx == 0:
                image_x = image_y
            else:
                cur_x = image_x.shape[3]
                image_x[:, :, :, -thx:cur_x, :] = (image_x[:, :, :, -thx:cur_x, :] + image_y[:, :, :, 0:thx, :]) / 2
                image_x = torch.cat([image_x, image_y[:, :, :, thx:x_block_size, :]], dim=3)
            xx = xx + 1

    image = image_x[:, :, 0:z_img, 0:y_img, 0:x_img]
    return image
