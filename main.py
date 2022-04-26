
from trainer import trainer
from evaluator import evaluator
from evaluator import eval_loss_plot
from test2 import test2
import argparse
from pathlib import Path
import torch
import yaml
#from model import model

def get_args_parser():
    parser = argparse.ArgumentParser('Set frcnn detector', add_help=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batchsize', default=3, type=int)
    parser.add_argument('--num_head',default=10,type=int)
    parser.add_argument('--atten_temp',default='defalt')########'defalt' or float
    parser.add_argument('--backbone',default='mobilenet')

    parser.add_argument('--model', default='FRCNN',
                        help='Choose the models FRCNN or SA-FRCNN or SA-FRCNN_v2 or SA-FRCNN_v3')
    parser.add_argument('--dataset_name', default='small_bdd',
                        help='Choose the dataset small_bdd or small_bdd_night bdd_val or hokkaido or test')

    parser.add_argument('--val_dataset_name', default='small_bdd_val',
                        help='Choose the dataset small_bdd or small_bdd_night bdd_val or hokkaido or test')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--train_model_path', default='',
                        help='input the train model path')     
    parser.add_argument('--img_path', default='',
                        help='input the test image path')                  

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval_loss',action='store_true')

    
    return parser


def main(args):
    
    
    #引数受け取り
    epochs=args.epochs
    lr=args.lr
    batchsize=args.batchsize
    output_dir=args.output_dir
    model_name=args.model
    num_head=args.num_head
    softmax_temp=args.atten_temp
    backbone=args.backbone
    

    with open('./dataset_path.yml', 'r',encoding="utf-8") as yml:
        data_path = yaml.load(yml, Loader=yaml.SafeLoader)
    
    dataset_name=args.dataset_name
    
    anno = data_path[dataset_name]["anno"]
    img = data_path[dataset_name]["img"]
    dataset_type = data_path[dataset_name]["dataset_type"]
    

    



    #データをロード
    #dataset_type='4class'

    data_ALL=[[anno,img]]

    #evalデータの設定
    #eva_img='C:/Research/dataset/BDD100K/images/100k/val'
    #eva_anno='C:/Research/dataset/BDD100K/annotation/val_small_bdd/val'

    val_dataset_name=args.val_dataset_name
    eva_anno=data_path[val_dataset_name]["anno"]
    eva_img=data_path[val_dataset_name]["img"]

    eval_data_ALL=[[eva_anno,eva_img]]

    if args.eval:
        train_model=args.train_model_path
        evaluator(data_ALL,dataset_type,batchsize,train_model)

        return

    if args.test:
        train_model=args.train_model_path
        img_path=args.img_path
        #test(dataset_type,train_model,img_path,output_dir)
        test2(dataset_type,train_model,batchsize,img_path,output_dir)

        return

    if args.eval_loss:
        train_model=args.train_model_path
        eval_loss_plot(data_ALL,dataset_type,batchsize,train_model)

        return












    print("Start training")
    #modelの読み込み
    from model import model
    #学習済みデータ
    train_model=args.train_model_path

    if train_model=='':
        print('first_train')
        model=model(dataset_type,model_name,num_head,softmax_temp,backbone)
    else:
        print(f'train_model:{train_model}')
        model=torch.load(train_model)

    #trainer(model,data_ALL,eval_data_ALL,dataset_type,epochs,lr,batchsize,model_name,output_dir)
    trainer(model,data_ALL,eval_data_ALL,dataset_type,args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FRCNN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)