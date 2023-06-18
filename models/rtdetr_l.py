'''
///////////////////////////////////////////////////////////////////////////
Code written by Pranav Durai on 06.06.2023 @ 21:50:12

Model: RTDeTR-L (Realtime Detection Transformer)

Framework: PyTorch 2.0
///////////////////////////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from detr_layers.hgstem import HGStem
from detr_layers.hgblock import HGBlock
from detr_layers.dwconv import DWConv
from detr_layers.aifi import AIFI
from detr_layers.conv import Conv
from detr_layers.concat import Concat
from detr_layers.repc3 import RepC3
from detr_layers.decoder import RTDETRDecoder

class RTDeTRL(nn.Module):
    def __init__(self, num_classes, scales):
        super(RTDeTRL, self).__init__()
        self.num_classes = num_classes

        self.backbone = nn.ModuleList([
            HGStem(3, int(48 * scales[1])),
            HGBlock(int(48 * scales[1]), int(128 * scales[1]), 6),
            DWConv(int(128 * scales[1]), 3, 2, 1, False),
            HGBlock(int(96 * scales[1]), int(512 * scales[1]), 6),
            DWConv(int(512 * scales[1]), 3, 2, 1, False),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, False),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, True),
            HGBlock(int(192 * scales[1]), int(1024 * scales[1]), 5, True, True),
            DWConv(int(1024 * scales[1]), 3, 2, 1, False),
            HGBlock(int(384 * scales[1]), int(2048 * scales[1]), 5, True, False)
        ])

        self.head = nn.ModuleList([
            Conv(int(256 * scales[1]), 256, 1, False),
            AIFI(int(1024 * scales[1]), 8),
            Conv(256, 256, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 1, False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 1, False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 256, 3, stride=2),
            Conv(256 + 256, 256, 3, stride=2),
            Conv(256 + 256, 256, 3, stride=2),
            RTDETRDecoder([256, 256, 256], self.num_classes)
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [3, 5, 7, 9]:
                features.append(x)

        for i, layer in enumerate(self.head):
            if isinstance(layer, nn.Upsample):
                x = layer(x)
                x = torch.cat((x, features.pop()), dim=1)
            else:
                x = layer(x)

        return x


# Create an instance of the RTDeTRL model
model = RTDeTRL(num_classes=80, scales={'l': [1.00, 1.00, 1024]})