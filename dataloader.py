'''
さらに高速化
背景のみの時の改善
スケール、バッジサイズを引数指定可能

'''
import numpy as np
#import pandas as pd
 
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
import yaml


class xml2list(object):
    
    def __init__(self, classes):
        self.classes = classes
        
    def __call__(self, xml_path):
        
        ret = []
        xml = ET.parse(xml_path).getroot()
        
        #たぶんいらない
        #for size in xml.iter("size"):     
        #    width = float(size.find("width").text)
        #    height = float(size.find("height").text)
    
        ###################################################################################################
        boxes = []
        labels = []
        zz=0
        
        for zz,obj in enumerate(xml.iter('object')):
            
            label = obj.find('name').text
           
            #print(label)
            ##指定クラスのみ
            #classes2= ['car', 'person', 'bike','motor','rider','truck', 'bus'] ##bdd100kをhokkaido対応させる応急処置  
            classes2=[] 
            if label in self.classes :
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.classes.index(label))
            elif label in classes2:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                if label=='rider':
                    #riderはperson判定
                    labels.append(classes2.index('person'))
                elif label=='truck' or label=='bus':
                    #truck,busはcar判定
                    labels.append(classes2.index('car')) 
                else:                   
                    labels.append(classes2.index(label))
            else:
                continue
            
        num_objs = zz +1

        ##BBOXが０の時がある、、
        #annotations = {'image':img, 'bboxes':boxes, 'labels':labels}
        anno = {'bboxes':boxes, 'labels':labels}

        return anno,num_objs
        ##################################################################################################




class MyDataset(torch.utils.data.Dataset):
        
    
        #def __init__(self, df, image_dir):
        def __init__(self,image_dir,xml_paths,scale,classes):
            
            super().__init__()
            self.image_dir = image_dir
            self.xml_paths = xml_paths
            self.image_ids = sorted(glob('{}/*.xml'.format(xml_paths)))
            self.scale=scale
            self.classes=classes
            
        def __getitem__(self, index):
    
            transform = transforms.Compose([
                                            transforms.ToTensor()
            ])
    
            # 入力画像の読み込み
            #image_id = self.image_ids[index]
            image_id=self.image_ids[index].split("\\")[-1].split(".")[0]
            image = Image.open(f"{self.image_dir}/{image_id}.jpg")
            
            
            ##################画像のスケール変換######################
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
            #########################################################
            image = transform(image)
        
            ###########################
            #classes = ['car', 'person', 'bicycle', 'motorbike']
            transform_anno = xml2list(self.classes)
            path_xml=f'{self.xml_paths}/{image_id}.xml'
            

            annotations,obje_num= transform_anno(path_xml)

            boxes = torch.as_tensor(annotations['bboxes'], dtype=torch.int64)
            labels = torch.as_tensor(annotations['labels'], dtype=torch.int64)

            #no-bbox
            if len(boxes)==0:
                
                iscrowd = torch.zeros((obje_num,), dtype=torch.int64)
                #area=[0]
                area = torch.as_tensor([[0]], dtype=torch.float32)
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels
                target["image_id"] = torch.tensor([index])
                target["area"] = area
                target["iscrowd"] = iscrowd
                #print(image_id)
                return image, target,image_id

            else:
                
                #bboxの縮小
                #print('縮小前:',boxes)
                boxes=boxes*ratio
                #print('縮小後:',boxes)
            
                area = (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2]-boxes[:, 0])
                area = torch.as_tensor(area, dtype=torch.float32)

                iscrowd = torch.zeros((obje_num,), dtype=torch.int64)

                #print(labels+1)
                #print(image_id)
                #print(area)
                #print(iscrowd)
                target = {}
                target["boxes"] = boxes
                target["labels"] = labels+1
                target["image_id"] = torch.tensor([index])
                target["area"] = area
                target["iscrowd"] = iscrowd
                #print(target)
                return image, target,image_id
        
        def __len__(self):
            #return self.image_ids.shape[0]
            return len(self.image_ids)
        
def dataloader (data,dataset_class,batch_size,scale=720,shuffle=True):
    itr=0
    for d in data:
        xml_paths=d[0]
        image_dir1=d[1]

        with open('./dataset_classes.yml', 'r',encoding="utf-8") as yml:
            data_classes_list = yaml.load(yml, Loader=yaml.SafeLoader)

        classes = data_classes_list[dataset_class]
        
        """
        if dataset_class =='bdd100k':
            classes = ['person', 'traffic light', 'train', 'traffic sign', 'rider', 'car', 'bike', 'motor', 'truck', 'bus']
        elif dataset_class == 'hokkaido' or  dataset_class == 'original_VOC':
            classes = ['car', 'bus', 'person', 'bicycle', 'motorbike', 'train']
        elif dataset_class == '4class':
            classes = ['car', 'person', 'bicycle', 'motorbike']
        else:
            print('Error : please choose dataset bdd100K or hokkaido')
            exit()
        """

        #dataset = MyDataset(df, image_dir1)
        dataset = MyDataset(image_dir1,xml_paths,scale,classes)
        

        

        #データのロード
        torch.manual_seed(2020)
        
        if itr == 0:
            train=dataset
        else:
            train=torch.utils.data.ConcatDataset([train,dataset])
        itr=itr+1
        
        
    

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn, pin_memory=True)#3
    #train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)#3

    return train_dataloader

    #データセットテスト用 
    #return train_dataloader,dataset
    



#テスト

#データの場所



bdd_val_xml="C:/Research/dataset/BDD100K/annotation/val"
bdd_val_img="C:/Research/dataset/BDD100K/images/100k/val"

data_ALL=[[bdd_val_xml,bdd_val_img]]
dataset_class='bdd100k'

#data_ALL=[[original_VOC_xml,original_VOC_img],[bdd_xml,bdd_img]]

#テスト兼ラベルの抽出(batchsize=1)でないとだめ
'''
t=dataloader_v6(data_ALL,dataset_class,3)
for images, targets,image_ids in t:
    
    
    for t in targets:
        targets = {k: v for k, v in t.items()}
        if 3 in targets['labels'] :
            print(image_ids[0])
            #exit()
'''
  




'''
###
#表示実験したい場合はdataloader_v6でdatasetを出力
t,dataset=dataloader_v6(data_ALL,dataset_class,3)
import matplotlib.pyplot as plt
import cv2
classes = ('__background__','car', 'person', 'bicycle', 'motorbike')
colors = ((0,0,0),(255,0,0),(0,255,0),(0,0,255),(100,100,100))
#image1, target,image_id = dataset[470]###null
image1, target,image_id = dataset[0]
#tensor2numpy
image1 = image1.to('cpu').detach().numpy().copy()
#print('------------------------')
image1=image1*255
image1=image1.transpose(1, 2, 0)
image1 = np.ascontiguousarray(image1, dtype=np.uint8)


#print(image1)
#image1 = image1.mul(255).permute(1, 2, 0).byte().numpy()
labels = target['labels'].cpu().numpy()
boxes = target['boxes'].cpu().numpy()
boxes=boxes.astype(np.int64)


for i,box in enumerate(boxes):   
    txt = classes[labels[i]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c = colors[labels[i]]
    cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), c , 2)
    cv2.rectangle(image1,(box[0], box[1] - cat_size[1] - 2),(box[0] + cat_size[0], box[1] - 2), c, -1)
    cv2.putText(image1, txt, (box[0], box[1] - 2), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)




plt.figure(figsize=(20,20))

plt.imshow(image1)

plt.show()

'''