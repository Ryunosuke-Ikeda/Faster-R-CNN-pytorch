
import torch
import torchvision
from torch import nn


class ResNetBackBone(nn.Module):
    '''
    torchvisionのresnetの中身を見てみると、conv1あたりの前処理の後に、特徴量抽出部分がありlayer1～4が該当します。ここの中間出力をバックボーンの出力として使います。
    　出力は4つのtensorで、それぞれ画像の大きさがwidth, heightが1/4, 1/8, 1/16, 1/32になっています。
    　また、resnet_typeによってスケール可能にしてあります。
    　出力チャンネル数はモデルのスケールによって違っており、以下の通りです。

    resnet18, 34は[64, 128, 256, 512] (左からout1,out2,out3,out4)
    50以上は[256, 512, 1024, 2048]
    https://deecode.net/?p=1226



    '''
    def __init__(self, resnet_type="resnet50", pretrained=True):
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet34":
            self.resnet_model = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet_model = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet_type == "resnet101":
            self.resnet_model = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet_type == "resnet152":
            self.resnet_model = torchvision.models.resnet152(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)  # width, heightは1/4
        out2 = self.resnet_model.layer2(out1)  # width, heightは1/8
        out3 = self.resnet_model.layer3(out2)  # width, heightは1/16
        out4 = self.resnet_model.layer4(out3)  # width, heightは1/32
        #return out1, out2, out3, out4
        return out3