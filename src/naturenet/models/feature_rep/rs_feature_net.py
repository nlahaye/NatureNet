

#Feature Pyramid Nets - Take channel count down from 50 to ~2

#FOr parameter inputs (Wave height, etc.) - Just do 1x1 convs to keep features as-is

import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm

from collections import OrderedDict

from naturenet.models.feature_rep.rs_feature_reduce import RSFeatureReduc


class RSFeatureNet(nn.Module):
    def __init__(self, in_chans, tile_size, mean, std):
        super(RSFeatureNet, self).__init__()
        self.in_chans = in_chans
        self.tile_size = tile_size
        #self.reduc = False

        self.add_module("reduc", RSFeatureReduc(in_chans, tile_size, mean, std))
        
        self.fpn_in_chans = (getattr(self, "reduc")).out_chans
        if self.fpn_in_chans >= 5:
             self.out_chans = 5
        else:
            self.out_chans = self.fpn_in_chans

        self.add_module("fpn", torchvision.ops.FeaturePyramidNetwork([self.fpn_in_chans,self.fpn_in_chans,self.fpn_in_chans], self.out_chans))



    def forward(self, x):
        x = torch.from_numpy(x) 
        #if torch.cuda.is_available():
        #    x = x.cuda()
        x1, x2, x3, grid_size = getattr(self, "reduc")(x)
  
        dct = OrderedDict()
        dct["x1"] = x1
        dct["x2"] = x2
        dct["x3"] = x3
        dct = getattr(self, "fpn")(dct)

        x1 = dct["x1"]
        x2 = dct["x2"]
        x3 = dct["x3"]

        if torch.cuda.is_available():
            x1 = x1.detach().cpu()
            x2 = x2.detach().cpu()
            x3 = x3.detach().cpu()


        return x1,x2,x3, grid_size




