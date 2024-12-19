import sys
sys.path.append("/home/crz/crz/Neuron_Segmentation2")
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from io1 import TESTDATASET,save_image
from DTAnet import DTAnet
from utils import cut_image,splice_image,cut_image_overlap,splice_image_overlap

torch.cuda.set_device(0)
class Test():
    def __init__(self,model_path,model_name,train_num,test_dataset_path,result_path,device):
        self.model_path = os.path.join(model_path,model_name)
        self.model_name = model_name
        self.train_num = train_num
        self.test_dataset_path = test_dataset_path
        self.result_path = result_path
        self.device = device

        dataset = TESTDATASET(self.test_dataset_path)
        self.test_data = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        self.model = DTAnet(in_dim=1,class_num=2,filter_num=32)
        model_NAME = self.model_name + '_' + str(self.train_num) + '.pkl'
        checkpoint = torch.load(os.path.join(self.model_path, model_NAME), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()

    def test(self,block_size,step):
        save_path = os.path.join(self.result_path, self.model_name + '_' + str(self.train_num))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for data in tqdm(self.test_data):
            image, filename = data
            image_size = image.shape[2:5]
            block_list,block_num,max_num = cut_image(image,block_size,step)
            output_list = []
            for i in range(block_num):
                image_block = block_list[i].to(self.device)
                predict_label = self.model(image_block)
                predict_label = predict_label.detach()
                output_list.append(predict_label)
            output = splice_image(output_list,block_num,max_num,image_size,step)
            self.Adjust_output(output,save_path,filename[0])

    def test_overlap(self,block_size,step):
        save_path = os.path.join(self.result_path, self.model_name + '_' + str(self.train_num))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for data in tqdm(self.test_data):
            image, filename = data
            image_size = image.shape[2:5]
            block_list,block_num,max_num = cut_image_overlap(image,block_size,step)
            output_list = []
            for i in range(block_num):
                image_block = block_list[i].to(self.device)
                #print(image_block.shape)
                image_block = image_block/image_block.max()
                #image_block = (image_block - image_block.min()) / (image_block.max() - image_block.min())
                predict_label = self.model(image_block)
                predict_label = predict_label.detach()
                output_list.append(predict_label)
            output = splice_image_overlap(output_list,block_num,max_num,image_size,step)
            self.Adjust_output(output,save_path,filename[0],mode='OVERLAP')

    def test_classify(self,block_size,step):
        save_path = os.path.join(self.result_path, self.model_name + '_' + str(self.train_num))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for data in tqdm(self.test_data):
            image, filename = data
            image_size = image.shape[2:5]
            block_list,block_num,max_num = cut_image(image,block_size,step)
            output_list = []
            for i in range(block_num):
                image_block = block_list[i].to(self.device)
                predict_class = self.model.class_forward(image_block)
                if torch.argmax(predict_class) == 0:
                    predict_label = image_block.detach()
                else:
                    predict_label = image_block.detach()+0.1
                output_list.append(predict_label)
            output = splice_image(output_list,block_num,max_num,image_size,step)
            self.Adjust_output(output,save_path,filename[0])
    def Adjust_output(self,image,save_path,filename,mode=None):
        if mode==None:
            save_path = os.path.join(save_path,'raw')
        elif mode == 'OVERLAP':
            save_path = os.path.join(save_path, 'overlap')
        elif mode == 'CLASS':
            save_path = os.path.join(save_path, 'classify')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_image(image*255, save_path, 'SEG_' + filename)

if __name__ == '__main__':
    print('test')
    MODEL_PATH = 'checkpoint'
    #MODEL_NAME = 'UNET3D'
    MODEL_NAME = 'S4_3'
    test_dataset_path = '../Neuron_Segmentation4/dataset/NG3'
    result_path = 'result'
    #test_dataset_path = 'dataset/Segment_dataset17302'
    #result_path = 'trainer1/result17302'
    block_size = [64,64,64]
    step = [64,64,64]
    overlap_step = [56,56,56]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda: ", device)

    tester = Test(MODEL_PATH,
                  MODEL_NAME,
                  train_num=30,
                  test_dataset_path = test_dataset_path,
                  result_path = result_path,
                  device=device)
    #tester.test(block_size,step)
    tester.test_overlap(block_size,overlap_step)
    #tester.test_classify(block_size,step)
    #tester.test_singel(block_size, step)