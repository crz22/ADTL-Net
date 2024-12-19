import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from io1 import TRAINDATASET,save_loss
from DTAnet import DTAnet
torch.cuda.set_device(2)

def Dice_loss(pre, lab):
    esp = 1
    # esp = 1e-5
    # esp = 0.01
    intersection = (pre * lab).sum(dim=[2, 3, 4])
    unionset = pre.sum(dim=[2, 3, 4]) + lab.sum(dim=[2, 3, 4])
    dice_loss = 1 - (2 * intersection + esp) / (unionset + esp)
    #print(dice_loss)
    return dice_loss.mean()
class TRAINER():
    def __init__(self,model_name,dataset_path,batch_size,EPOCH,lr,lr_decay_epoch=10,save_model_num=10):
        dataset = TRAINDATASET(dataset_path)
        self.train_dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.EPOCH = EPOCH
        self.iter = len(self.train_dataset)

        self.save_path = 'checkpoint/' + model_name
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_name = model_name
        self.save_model_num = save_model_num

        self.model = DTAnet(in_dim=1, class_num=2, filter_num=32).to(device)
        self.model.train()
        self.lr = lr
        self.lr_decay_mode = 'StepLR'
        self.decay_epoch = lr_decay_epoch
        self.decay_weight = 0.1
        self.Net_opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.clamd = 2
        self.slamd = 1
        self.step_loss = {}

    def caculate_class_loss(self,predict,label):
        weight = label.clone()
        weight[:,1] = 0
        weight = weight + 1
        weight.to(device)
        weight_BCE = nn.BCELoss(weight=weight)
        class_loss = weight_BCE(predict, label)
        self.step_loss['closs'] = self.step_loss['closs'] + class_loss.item() / self.iter
        return class_loss

    def caculate_seg_loss(self,predict,label):
        # Dice loss
        #dice_loss = Dice_loss(predict, label)
        #self.step_loss['Dice'] = self.step_loss['Dice'] + dice_loss.item() / self.iter
        dice_loss = 0
        # Bce loss
        BCE = nn.BCELoss()
        bce_loss = BCE(predict,label)*10
        self.step_loss['Bce'] = self.step_loss['Bce'] + bce_loss.item() / self.iter

        # Weighted BCE loss
        # weight = label * 10 + (1 - label)
        # weight = weight/torch.sum(weight,dim=[2,3,4],keepdim=True)
        # WBCE = nn.BCELoss(weight=weight)
        # wbce_loss = WBCE(predict_label, label)
        # self.step_loss['WBce'] = self.step_loss['WBce'] + wbce_loss.item() / self.iter

        seg_loss = dice_loss + bce_loss
        self.step_loss['sloss'] = self.step_loss['sloss'] + seg_loss.item() / self.iter
        # print(dice_loss,bce_loss)
        return seg_loss

    def train_step(self):
        self.step_loss.clear()
        self.step_loss.setdefault("sloss", 0)
        self.step_loss.setdefault("closs", 0)
        self.step_loss.setdefault("Dice", 0)
        self.step_loss.setdefault("Bce", 0)
        for data in tqdm(self.train_dataset):
            image, slabel,clabel,_ = data
            image, slabel, clabel = image.to(device), slabel.to(device), clabel.to(device)
            # train class task
            predict_class = self.model.class_forward(image)
            class_loss = self.caculate_class_loss(predict_class,clabel)

            # train segment task
            predict_seg = self.model.segment_forward(image, clabel)
            seg_loss = self.caculate_seg_loss(predict_seg,slabel)

            loss = class_loss*self.clamd + seg_loss*self.slamd

            # print(seg_loss,seg_loss_dice,seg_loss_bce)
            self.Net_opt.zero_grad()
            loss.backward()
            self.Net_opt.step()

            #break

    def train(self):
        for epoch in range(self.EPOCH):
            print(epoch,'/',self.EPOCH)
            self.train_step()
            # clear loss.txt
            if epoch == 0 and os.path.exists(self.save_path + '/loss.txt'):
                os.remove(self.save_path + '/loss.txt')
            save_loss(self.save_path, self.step_loss)

            if (epoch + 1) % self.save_model_num == 0:
                self.save_model(epoch + 1)
            self.update_LR(epoch + 1)

    def update_LR(self, epoch):
        if self.lr_decay_mode == 'StepLR':
            if epoch % self.decay_epoch == 0:
                for params in self.Net_opt.param_groups:
                    params['lr'] = params['lr'] * self.decay_weight
        else:
            assert 0, "lr decay initial False"
        print("Lr: ", self.Net_opt.state_dict()['param_groups'][0]['lr'])

    def save_model(self, epoch):
        print("save model")
        save_path = self.save_path + '/' + self.model_name + '_' + str(epoch) + '.pkl'
        torch.save(self.model.state_dict(), save_path)



if __name__ == '__main__':
    train_dataset_path = 'dataset/train_dataset1_CW'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = TRAINER(model_name='S4_3',
                      dataset_path=train_dataset_path,
                      batch_size=16,
                      EPOCH=60,
                      lr=0.001,
                      lr_decay_epoch=10,
                      save_model_num=10)
    trainer.train()
