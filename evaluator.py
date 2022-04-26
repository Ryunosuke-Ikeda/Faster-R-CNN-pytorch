#元val_score_v3
import numpy as np
#import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
import os
import matplotlib.pyplot as plt
from natsort import natsorted
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from dataloader import dataloader
from eval_loss import eval_one_epoch
from model import model


def evaluator(data_ALL,dataset_type,batch_size,train_model):
    val_dataloader=dataloader(data_ALL,dataset_type,batch_size,shuffle=False)

    #####データのはいったファイルを選択##################
    #複数データ
    #INPUT_DIR='./train_model/small_bdd_night/'
    model_files = natsorted(glob(os.path.join(train_model, '*.pt')))
    #print(model_files)

    #単一データ
    #model_files= [train_model]

    if len(model_files)==0:
        print('The train_model argument should be a folder name, not a \'.pt\' file name.')
        exit()

    #print(model_files)
    #print('--------------------------------')

    print(f'dataset_type:{dataset_type}')
    print(f'testdata: {data_ALL}')
    print(f'batch_size {batch_size}')
    #################################################
    


    #GPUのキャッシュクリア
    import torch
    torch.cuda.empty_cache()
    from engine import train_one_epoch, evaluate
    import utils
    print(model_files)
    for filename in model_files:
        model=torch.load(filename)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model.cuda()

        print('-------------',filename,'--------------------')

        # evaluate on the test dataset
        evaluate(model, val_dataloader, device=device)



def eval_loss_plot(data_ALL,dataset_type,batch_size,train_model):
    eval_dataloader=dataloader(data_ALL,dataset_type,batch_size,shuffle=False)

    #####データのはいったファイルを選択##################
    #複数データ
    #INPUT_DIR='./train_model/small_bdd_night/'
    model_files = natsorted(glob(os.path.join(train_model, '*.pt')))
    #print(model_files)

    if len(model_files)==0:
        print('The train_model argument should be a folder name, not a \'.pt\' file name.')
        exit()

    
    #単一データ
    #model_files= [train_model]
    print(model_files)
    print('--------------------------------')

    print(f'dataset_type:{dataset_type}')
    print(f'testdata: {data_ALL}')
    print(f'batch_size {batch_size}')
    #GPUのキャッシュクリア
    import torch
    torch.cuda.empty_cache()
    from engine import train_one_epoch, evaluate
    import utils
    


    eval_loss_list=[]
    loss_class=[]
    loss_box_reg=[]
    loss_obje=[]
    loss_rpn_reg=[]
    for i,filename in enumerate(model_files):
        model=torch.load(filename)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        model.cuda()

        print('eval')
        print(f'model:{filename}')
        loss_eval,loss1,loss2,loss3,loss4=eval_one_epoch(model, eval_dataloader, device, epoch=1, print_freq=500)
        eval_loss_list.append(np.mean(loss_eval))
        loss_class.append(np.mean(loss1))
        loss_box_reg.append(np.mean(loss2))
        loss_obje.append(np.mean(loss3))
        loss_rpn_reg.append(np.mean(loss4))

        if i>1:
            #グラフの描画
            fig=plt.figure()
            plt.plot(range(len(eval_loss_list)), eval_loss_list, 'g-', label='val_loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid()
            #plt.show()
            fig.savefig(f"{train_model}/val_loss.png")

            fig=plt.figure()
            plt.plot(range(len(loss_class)), loss_class, 'r-', label='loss_class')
            plt.plot(range(len(loss_class)), loss_box_reg, 'g-', label='loss_box_reg')
            plt.plot(range(len(loss_class)), loss_obje, 'b-', label='loss_obje')
            plt.plot(range(len(loss_class)), loss_rpn_reg, 'y-', label='loss_rpn_reg')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid()
            #plt.show()
            fig.savefig(f"{train_model}/val_4loss.png")




