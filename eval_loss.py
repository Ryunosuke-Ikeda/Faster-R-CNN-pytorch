import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import copy



def eval_one_epoch(model, data_loader, device, epoch, print_freq):

    model_eval = copy.deepcopy(model)


    model_eval.train()
    #metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #header = 'Epoch: [{}]'.format(epoch)


    loss_epo=[]########

    loss_class=[]
    loss_box_reg=[]
    loss_giou=[]
    loss_obje=[]
    loss_rpn_reg=[]
    #for images, targets in metric_logger.log_every(data_loader, print_freq, header):######dataloaderv4
    #for images, targets,image_ids in metric_logger.log_every(data_loader, print_freq, header):#####dataloaderv3
    for i,batch in enumerate(data_loader):

        images, targets, image_ids = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        ######
        
        #loss_dict = model_eval(images, targets)
        with torch.no_grad():
            try:
                loss_dict = model_eval(images, targets)

            except ValueError:
                #print(image_ids)
                pass
            else:
                #losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)

        
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                
            
                loss_value = losses_reduced.item()

                loss_epo.append(loss_value)#########

                
                #各lossの出力
                loss_value_c=loss_dict_reduced['loss_classifier'].item()
                loss_value_b=loss_dict_reduced['loss_box_reg'].item()
                loss_value_o=loss_dict_reduced['loss_objectness'].item()
                loss_value_r=loss_dict_reduced['loss_rpn_box_reg'].item()
                #loss_value_g=loss_dict_reduced['loss_giou'].item()

                loss_class.append(loss_value_c)
                loss_box_reg.append(loss_value_b)
                loss_obje.append(loss_value_o)
                loss_rpn_reg.append(loss_value_r)
                #loss_giou.append(loss_value_g)

                if (i+1) % 200== 0:
                    print(f"epoch #{epoch} Iteration #{i+1} loss: {loss_value}")

            
    del model_eval
    return loss_epo,loss_class,loss_box_reg,loss_obje,loss_rpn_reg