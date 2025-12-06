

#Feature Pyramid Nets - Take channel count down from 50 to ~2

#FOr parameter inputs (Wave height, etc.) - Just do 1x1 convs to keep features as-is

import torch
import torch.nn as nn

from naturenet.models.feature_rep.rs_feature_reduce import RSFeatureReduc


class RSFeatureReduc(nn.Module):
    def __init__(self, in_chans, tile_size, mean, std):

class RSFeatureNet(nn.Module):
    def __init__(self, in_chans, tile_size, mean, std):
        self.in_chans = in_chans
        self.reduc = False

        self.add_module("reduc", RSFeatureReduc(in_chans, tile_size, mean, std))
        
        self.fpn_in_chans = (getattr(self, "reduc")).out_chans
        if self.fpn_in_chans >= 5:
             self.out_chans = 5
        else:
            self.out_chans = self.fpn_in_chans

        self.add_module("fpn", torchvision.ops.FeaturePyramidNetwork([fpn_in_chans,fpn_in_chans,fpn_in_chans], self.out_chans))



    def forward(self, x):
        x1, x2, x3, grid_size = getattr(self, "reduc")(x)
        x1,x2,x3 = getattr(self, "fpn")([x1,x2,x3])

        return x1,x2,x3, grid_size




