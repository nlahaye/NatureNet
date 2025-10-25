

import torch
import torch.nn as nn

#1x1 Convolutions per pixel to reduce to < 50

class RSFeatureReduc(nn.Module):
    def __init__(self, in_chans, tile_size):
        super(RSFeatureReduc, self).__init__()
        od = in_chans
        self.in_chans = in_chans

        self.add_module("batch_norm", nn.BatchNorm2d(od))

        layer_ind = 1
        current_chans = od

 
        if in_chans > 50:
            while current_chans > 50:
                self.add_module("reduc" + str(i), nn.Conv2d(od, int(od/5), kernel_size=1))
                self.add_module("reduc" + str(i) + "_act" + str(j), nn.LeakyReLU(0.1, inplace=True))

                self.add_module("reduc" + str(i) + "_2", nn.Conv2d(od, int(od/5), kernel_size=3, stride=2))
                self.add_module("reduc" + str(i) + "2_act" + str(j), nn.LeakyReLU(0.1, inplace=True))

                self.add_module("reduc" + str(i) + "_3", nn.Conv2d(od, int(od/5), kernel_size=5, stride=4))
                self.add_module("reduc" + str(i) + "3_act" + str(j), nn.LeakyReLU(0.1, inplace=True))
 

                od = int(od/5)
                layer_ind = layer_ind+1

            self.n_layers = layer_ind

        else:
            self.add_module("reduc1", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=1))
            self.add_module("reduc1_act" + str(j), nn.LeakyReLU(0.1, inplace=True))

            self.add_module("reduc1_2", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, stride=2))
            self.add_module("reduc12_act" + str(j), nn.LeakyReLU(0.1, inplace=True))

            self.add_module("reduc1_3", nn.Conv2d(self.in_chans, self.in_chans, kernel_size=5, stride=4))
            self.add_module("reduc13_act" + str(j), nn.LeakyReLU(0.1, inplace=True))
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
        x = getattr(self, "batch_norm")(x)

        tiled_data = patchify(x, (self.tile_size, self.tile_size, x.shape[2]), self.tile_size)
        tiled_data = torch.flatten(tiled_data, start_dim=0, end_dim = 2)
        print(tiled_data.shape)

        x1_out = []
        x2_out = []
        x3_out = []
        for j in range(1,self.n_layers+1):
            for k in range(tiled_data.shape)
                x = getattr(self, "reduc" + str(i))(tiled_data[k])
                x = getattr(self, "reduc" + str(i) "_act")(x)

                x2 = getattr(self, "reduc" + str(i) + "_2")(tiled_data[k])
                x2 = getattr(self, "reduc" + str(i) + "2_act")(x2)

                x3 = getattr(self, "reduc" + str(i) + "_3")(tiled_data[k])
                x3 = getattr(self, "reduc" + str(i) + "3_act")(x3)

                x1_out.append(x)
                x2_out.append(x2)
                x3_out.append(x3)


        return x1_out, x2_out, x3_out



