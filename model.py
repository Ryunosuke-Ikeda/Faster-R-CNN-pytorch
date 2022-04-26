import numpy as np
#import pandas as pd
 
from PIL import Image
from glob import glob
import xml.etree.ElementTree as ET 
 
import torch
import torchvision
#import vision.torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from resnet_backbone import ResNetBackBone
import yaml


def model (dataset_type,model_name,backbone_model):
    #モデルの定義


    import torchvision
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    
    if backbone_model == "mobilenet":
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
    elif backbone_model ==  "resnet":
        backbone=ResNetBackBone(resnet_type="resnet50")
        backbone.out_channels = 1024
    else :
        print('Error: please choose backbone mobilenet or resnet')
        exit()
    
    
    


    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
   
    #anchor_generator = AnchorGenerator(sizes=((64, 128, 256),),
    #                                aspect_ratios=((0.5, 1.0, 2.0),))
 
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.

    
    
    
        
    # put the pieces together inside a FasterRCNN model


    with open('./dataset_classes.yml', 'r',encoding="utf-8") as yml:
        data_classes_list = yaml.load(yml, Loader=yaml.SafeLoader)
    classes = data_classes_list[dataset_type]

    num_classes=len(classes)+1
    

    if model_name=='FRCNN':
    
    
        model = FasterRCNN(backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator=anchor_generator)
    
    else:
        print('Error: Choose the model name ')
        exit()

                  
    print("num_classes:",num_classes)

    
  
    return model

