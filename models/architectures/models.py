# @copyright CEA-LIST/DIASI/SIALV/LVA (2020)
# @author CEA-LIST/DIASI/SIALV/LVA <quentin.bouniot@cea.fr>
# @license CECILL

import torch.nn as nn
from .resnet import resnet50

class Classification(nn.Module):

    def __init__(self, class_num, pretrained=True):
        super(Classification, self).__init__()
        backbone = resnet50(pretrained=pretrained)

        backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        backbone.fc = nn.BatchNorm1d(2048)        
        self.backbone = backbone

        self.fc = nn.Linear(2048, class_num, bias = True)

    def forward(self, x, training=True):
        x = self.backbone(x)
        
        if not training:
            return x
        
        x = self.fc(x)
        
        return x
    
class Triplet(nn.Module):
    
    def __init__(self, embedding_size, pretrained=True):
        super(Triplet, self).__init__()
        backbone = resnet50(pretrained=pretrained)
        
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone = backbone
        
        self.backbone.fc = nn.Sequential(
                            nn.BatchNorm1d(self.backbone.fc.in_features),
                            nn.ReLU(),
                            nn.Linear(self.backbone.fc.in_features, embedding_size))
        
    def forward(self, x):
        x = self.backbone(x)
        
        return x
