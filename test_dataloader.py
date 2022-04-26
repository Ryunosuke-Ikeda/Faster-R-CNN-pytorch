'''
さらに高速化
背景のみの時の改善
スケール、バッジサイズを引数指定可能

'''
import numpy as np 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
import cv2
 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import TensorDataset
import os
import time

#データの場所


bdd_test_xml="../../dataset/bdd100K_to_VOC/train_test"
bdd_xml="../../dataset/bdd100K_to_VOC/train"
bdd_img="D:/bdd100k/bdd100k/bdd100k/images/100k/train"

bdd_val_xml="../../dataset/bdd100K_to_VOC/val/"
bdd_val_img="D:/bdd100k/bdd100k/bdd100k/images/100k/val"

hokkaido_test_xml="../../dataset/hokkaido_test/val_test/annotations/"
hokkaido_test_img="../../dataset/hokkaido_test/val_test/img"

original_VOC_xml="../../dataset/original_VOC/Annotations"
original_VOC_img="../../dataset/original_VOC/JPEGimages/"
dataset_class='4class'




image_dir_ALL=bdd_img
xml_paths_train=bdd_test_xml


class MyDataset(torch.utils.data.Dataset):
        def __init__(self,image_dir,scale):
            
            super().__init__()
            self.image_dir = image_dir
            self.image_ids = sorted(glob('{}/*'.format(image_dir)))
            self.scale=scale
        
            
        def __getitem__(self, index):
    
            transform = transforms.Compose([
                                            transforms.ToTensor()
            ])
            # 入力画像の読み込み
            image_id=self.image_ids[index].split("\\")[-1].split(".")[0]
            image = Image.open(self.image_ids[index])
            
            
            ##################画像のスケール変換######################
            '''
            t_scale_tate=self.scale ##目標のスケール(縦)
            #縮小比を計算
            ratio=t_scale_tate/image.size[1]
            ##目標横スケールを計算
            t_scale_yoko=image.size[0]*ratio
            t_scale_yoko=int(t_scale_yoko)
            
            #print('縮小前:',image.size)
            #print('縮小率:',ratio)
            #リサイズ
            image = image.resize((t_scale_yoko,t_scale_tate))
            #print('縮小後:',image.size)
            '''
            #########################################################
            image = transform(image)
        
            return image,image_id
        
        def __len__(self):
            return len(self.image_ids)
        
def test_dataloader (data,batch_size,scale=720,shuffle=False):
    itr=0
    for d in data:
        image_dir1=d

        dataset = MyDataset(image_dir1,scale)

        #データのロード
        torch.manual_seed(2020)
        
        if itr == 0:
            train=dataset
        else:
            train=torch.utils.data.ConcatDataset([train,dataset])
        itr=itr+1
        

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    
    test_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn, pin_memory=True)#3
    

    return test_dataloader

