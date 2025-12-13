

import torch
import torchvision
import torch.nn as nn

import numpy as np

from tqdm import tqdm

from skimage.util import view_as_windows

#1x1 Convolutions per pixel to reduce to < 50


class MultiSourceRSFeatureReduc(nn.Module):
    def __init__(self, in_chans):
        super(MultiSourceRSFeatureReduc, self).__init__()
        od = in_chans
        self.in_chans = in_chans
 

        layer_ind = 1
        current_chans = od

        self.out_chans = in_chans

        while current_chans > 10:
            self.add_module("reduc" + str(layer_ind), torchvision.ops.FeaturePyramidNetwork(od, int(od/2)))

            self.out_chans = od
            od = int(od/2)
            layer_ind = layer_ind+1

        self.n_layers = layer_ind


    def forward(self, x):
        for j in range(1,self.n_layers+1):
             x = getattr(self, "reduc" + str(j))(x)
        return x


class RSFeatureReduc(nn.Module):
    def __init__(self, in_chans, tile_size, mean, std):
        super(RSFeatureReduc, self).__init__()
        od = in_chans
        self.in_chans = in_chans
        self.mean = mean
        self.std = std
        self.tile_size = tile_size

        #Assuming image has been reconstructed before being passed through


        layer_ind = 1
        current_chans = od
        self.out_chans = self.in_chans
 
        if in_chans > 50:
            while current_chans > 50:
                self.add_module("reduc" + str(layer_ind), nn.Conv2d(od, int(od/5), kernel_size=1))
                self.add_module("reduc" + str(layer_ind) + "_act", nn.LeakyReLU(0.1, inplace=True))

                self.add_module("reduc" + str(layer_ind) + "_2", nn.Conv2d(od, int(od/5), kernel_size=2, stride=2))
                self.add_module("reduc" + str(layer_ind) + "2_act", nn.LeakyReLU(0.1, inplace=True))

                self.add_module("reduc" + str(layer_ind) + "_3", nn.Conv2d(od, int(od/5), kernel_size=4, stride=4))
                self.add_module("reduc" + str(layer_ind) + "3_act", nn.LeakyReLU(0.1, inplace=True))
 

                self.out_chans = od

                od = int(od/5)
                layer_ind = layer_ind+1
            self.n_layers = layer_ind

        else:
            self.add_module("reduc1", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=1))
            self.add_module("reduc1_act", nn.LeakyReLU(0.1, inplace=True))

            self.add_module("reduc1_2", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=2, stride=2))
            self.add_module("reduc12_act", nn.LeakyReLU(0.1, inplace=True))

            self.add_module("reduc1_3", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=4, stride=4))
            self.add_module("reduc13_act", nn.LeakyReLU(0.1, inplace=True))
            self.n_layers = 1 
           

        self.out_chans = od

        self.initialize_weights()


    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)


    def forward(self, x):
        i = 0
        #x = getattr(self, "batch_norm")(x)
        #Tile 0,0 is top left of image, tile N,N is bottom right
        torchvision.transforms.functional.normalize(x, self.mean, self.std, inplace=True)
        x = x.numpy()
        #Batch x Channel x Y x X

        tiled_data = torch.squeeze(torch.from_numpy(view_as_windows(x, (x.shape[0], x.shape[1], self.tile_size, self.tile_size),\
            (x.shape[0], x.shape[1], self.tile_size, self.tile_size)).astype(np.float32)))

        first_ind = -4
        last_ind = -3
        start_dim = 0
        end_dim = 1
        if tiled_data.ndim == 5:
            first_ind = -5
            last_ind = -4
        grid_size = [tiled_data.shape[first_ind], tiled_data.shape[last_ind]]
        tiled_data = torch.flatten(tiled_data, start_dim=start_dim, end_dim = end_dim) #Flattening number of samples (H,W,Chan)
        if last_ind == -3:
            tiled_data = torch.unsqueeze(tiled_data, 1)

        if torch.cuda.is_available():
            tiled_data = tiled_data.cuda()

        x1_out = []
        x2_out = []
        x3_out = []
        for j in range(1,self.n_layers+1):
            for k in tqdm(range(tiled_data.shape[0])):
                x = getattr(self, "reduc" + str(j))(tiled_data[k])
                x = getattr(self, "reduc" + str(j) + "_act")(x)

                x2 = getattr(self, "reduc" + str(j) + "_2")(tiled_data[k])
                x2 = getattr(self, "reduc" + str(j) + "2_act")(x2)

                x3 = getattr(self, "reduc" + str(j) + "_3")(tiled_data[k])
                x3 = getattr(self, "reduc" + str(j) + "3_act")(x3)
 
                if j == self.n_layers:
                    x1_out.append(x)
                    x2_out.append(x2)
                    x3_out.append(x3)

        if torch.cuda.is_available():
            tiled_data = tiled_data.detach().cpu()
 

        return torch.stack(x1_out, dim=0), torch.stack(x2_out, dim=0), torch.stack(x3_out, dim=0), grid_size



