import os
from libtiff import TIFF
import numpy as np
import torch
from torch.utils.data import Dataset

def read_image(image_path):
    images = []
    tif = TIFF.open(image_path, mode='r')
    for image in tif.iter_images():
        images.append(image)
    return np.array(images)

def save_image(img, filepath, image_name):
    img = img.squeeze().squeeze().cpu()
    img = img.detach().numpy()

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    filepath = filepath + '/' + image_name
    tif = TIFF.open(filepath, mode='w')
    num = img.shape[0]

    for i in range(num):
        tif.write_image(((img[i]).astype(np.uint8)), compression=None)
    tif.close()
    return

class TRAINDATASET(Dataset):
    def __init__(self,dataset_path):
        super(TRAINDATASET, self).__init__()
        self.image_path = dataset_path + '/image'
        self.label_path = dataset_path + '/label'
        self.filename = os.listdir(self.image_path)
        #self.filename = [file for file in self.filename if 'back' not in file]

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        image = read_image(os.path.join(self.image_path, self.filename[item]))/255.0
        label = read_image(os.path.join(self.label_path, self.filename[item]))/255.0
        #image = image/image.max()
        image = self.numpy_to_tensor(image)
        label = self.numpy_to_tensor(label)
        class_label = self.get_class_label(self.filename[item])
        #class_label = 0
        return image, label, class_label,self.filename[item]

    def numpy_to_tensor(self, image):
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image, 0)
        return image

    def get_class_label(self,filename):
        '''
        if 'WS' in filename:
            return torch.tensor([1,0,0],dtype=torch.float32)
        elif 'SN' in filename:
            return torch.tensor([0,1,0],dtype=torch.float32)
        else:
            return torch.tensor([0,0,1],dtype=torch.float32)#2
        '''
        if 'WS' in filename:
            return torch.tensor([1,0],dtype=torch.float32)
        elif 'SN' in filename:
            return torch.tensor([0,1],dtype=torch.float32)
        else:
            return torch.tensor([0,1],dtype=torch.float32)#2

class TESTDATASET(Dataset):
    def __init__(self,dataset_path):
        super(TESTDATASET, self).__init__()
        self.dataset_path = dataset_path
        self.image_path = dataset_path + '/image'
        self.filename = os.listdir(self.image_path)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        image = read_image(os.path.join(self.image_path, self.filename[item],self.filename[item]))
        image = self.numpy_to_tensor(image)
        #image= self.numpy_to_tensor(image/image.max())
        return image, self.filename[item]

    def numpy_to_tensor(self,image):
        image = torch.from_numpy(image).float()
        image = torch.unsqueeze(image,0)
        return image

def save_loss(save_path,loss_list):
    txt_path = save_path+'/loss.txt'
    with open(txt_path, 'a') as f:
        #Write loss name
        if os.path.getsize(txt_path) == 0:
            for key, value in loss_list.items():
                f.write(key)
                f.write(' ')
            f.write('\n')
        #Write loss value
        for key, value in loss_list.items():
            print(key, value)
            f.write(str(value))
            f.write(' ')
        f.write('\n')
        f.close()