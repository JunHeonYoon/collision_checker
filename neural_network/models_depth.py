import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, layer_sizes, batch_size):
        nn.Module.__init__(self)
        
        def init_weights(m):
            # print(m)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                # print(m.weight)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            # self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            # print('i:',i)
            # print(len(layer_sizes))
            # print(in_size, out_size)
            if i+2 < len(layer_sizes):
                # self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(batch_size))
                self.MLP.add_module(
                    # name="D{:d}".format(i), module=nn.Dropout(0.5 -(i) * 0.15))
                    name="D{:d}".format(i), module=nn.Dropout(0.5))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            else:
                # print('sigmoid!')
                self.MLP.add_module(name="sigmoid", module=nn.Tanh())

        self.MLP.apply(init_weights)

    def forward(self, x):
        x = self.MLP(x)
        
        return x

class CollNet(nn.Module):
    def __init__(self, fc_layer_sizes, batch_size, latent_size, device):
        
        nn.Module.__init__(self)

        assert type(latent_size) == int

        self.latent_size = latent_size

        self.local_occuapncy_vae = VAE(latent_size, device)
        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size)

    def forward(self, x_q, x_depth):
        """
        x: input nerf_q(21), depth(1048576:4 X 512 X 512)
        """
        recon_x, mean, log_var, z = self.local_occuapncy_vae(x_depth)
        x_fc = torch.cat([x_q,z], dim=1)
        y = self.fc(x_fc)
        
        return y, x_depth, recon_x, mean, log_var
        

