
import numpy as np
#import pandas as pd
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import time
import datetime
#import pytz
import matplotlib.pyplot as plt
#dataloader.pyとmodel.pyからインポート
from dataloader import dataloader
import yaml

#def trainer(model,data_ALL,eval_data_ALL,dataset_type,epochs,lr,batchsize,model_name,output_dir):
def trainer(model,data_ALL,eval_data_ALL,dataset_type,args):
    start = time.time()

    epochs=args.epochs
    lr=args.lr
    batchsize=args.batchsize
    output_dir=args.output_dir
    model_name=args.model
    num_head=args.num_head
    softmax_temp=args.atten_temp
    backbone=args.backbone
    


    #実行環境の表示
    #print(file_name)
    print(f'train_data:{data_ALL}')
    print(f'dataset_class:{dataset_type}')
    print(f'epoch:{epochs}')
    print(f'batchsize:{batchsize}')
    print(f'lr:{lr}') 
    print("BackBone:", backbone)
    print(f'model name: {model_name}')
    #print(model)

    folder_dir=f'./log/{output_dir}/{model_name}_{datetime.date.today()}'

    if not os.path.exists(folder_dir):  # ディレクトリが存在しない場合、作成する。
            os.makedirs(folder_dir)

    with open(f'{folder_dir}/opt.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    



    train_dataloader=dataloader(data_ALL,dataset_type,batchsize)

    
    eval_dataloader=dataloader(eval_data_ALL,dataset_type,1)


    ##学習
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)##オリジナル

    #GPUのキャッシュクリア
    torch.cuda.empty_cache()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model.cuda()##


    ##ここから
    from engine import train_one_epoch, evaluate
    from eval_loss import eval_one_epoch
    import utils
    # let's train it for 10 epochs
    num_epochs = epochs


    ##データの読み込み時間計測
    elapsed_time = time.time() - start
    print ("Dataload_time:{0}".format(elapsed_time) + "[sec]")

    #なんか早くなるらしい
    #torch.backends.cudnn.benchmark = True

    loss_list=[]
    eval_loss_list=[]

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        print('train')
        loss=train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        loss_list.append(np.mean(loss))

        print('eval')
        loss_eval,_,_,_,_=eval_one_epoch(model, eval_dataloader, device, epoch, print_freq=500)
        eval_loss_list.append(np.mean(loss_eval))



        

        #5epずつモデルを保存
        if epoch%10==0:
            torch.save(model, f'{folder_dir}/{epoch+1}.pt')

        #10epでグラフ表示
        if epoch%10==0 and epoch!=0:
            #グラフの描画
            fig=plt.figure()
            plt.plot(range(len(loss_list)), loss_list, 'r-', label='train_loss')
            plt.plot(range(len(eval_loss_list)), eval_loss_list, 'g-', label='val_loss')
            plt.legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.grid()
            #plt.show()
            #fig.savefig(f"{folder_dir}/{model_name}_{epoch+1}ep.png")
            fig.savefig(f"{folder_dir}/{model_name}.png")        

            

    torch.save(model, f'{folder_dir}/{model_name}.pt')

    


    elapsed_time = time.time() - start
    print ("train_time:{0}".format(elapsed_time) + "[sec]")


    #グラフの描画
    fig=plt.figure()
    plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
    plt.plot(range(len(eval_loss_list)), eval_loss_list, 'g-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    #plt.show()
    fig.savefig(f"{folder_dir}/{model_name}.png")

    with open(f'{folder_dir}/{model_name}.txt', 'w') as f:
        #print(model_name, file=f)
        print(f'train_data:{data_ALL}', file=f)
        print(f'dataset_class:{dataset_type}',file=f)
        print(f'epoch:{epochs}', file=f)
        print(f'batchsize:{batchsize}', file=f)
        print(f'num_head:{num_head}', file=f)
        print(f'atten_softmax_temp:{softmax_temp}',file=f)
        print(f'lr:{lr}', file=f) 
        print ("train_time:{0}".format(elapsed_time) + "[sec]",file=f)
        print(f'model name: {model_name}',file=f)
        print('model',file=f)
        print(model,file=f)
        print('##loss##',file=f)
        print(loss_list,file=f)
        print('##eval_loss##',file=f)
        print(eval_loss_list,file=f)


#trainer(model,data_ALL,dataset_type,epochs,lr,batchsize)