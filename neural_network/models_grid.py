import torch
import torch.nn as nn
import numpy as np
from math import floor

def convShape(in_size, kernel_size, stride, padding):
    return int(floor( (in_size + 2*padding - kernel_size) / stride + 1 ))
def convTransShape(in_size, kernel_size, stride, padding):
    return int( (in_size - 1) * stride - 2*padding + kernel_size )

class Encoder(nn.Module):
    def __init__(self, input_data_shape, encoder_layer, latent_size):
        nn.Module.__init__(self)
        self.encoder_layer = encoder_layer
        self.data_shape = [input_data_shape]
        self.MLP = nn.ModuleList()

        for layer_idx, layer in enumerate(self.encoder_layer):
            if layer["type"] == "Conv":
                assert self.data_shape[layer_idx]["channel"] == layer["in_channel"]
                self.MLP.append(nn.Conv3d(in_channels=layer["in_channel"],
                                          out_channels=layer["out_channel"],
                                          kernel_size=layer["kernel_size"],
                                          stride=layer["stride"],
                                          padding=0,
                                          bias=True))
                

                data_shape = {"channel": layer["out_channel"],
                              "shape"  : [convShape(e, layer["kernel_size"], layer["stride"], layer["padding"]) 
                                         for e in self.data_shape[layer_idx]["shape"]]}

                self.data_shape.append(data_shape)

            elif layer["type"] == "Relu":
                self.MLP.append(nn.ReLU())
                data_shape = {"channel": self.data_shape[layer_idx]["channel"],
                              "shape"  : self.data_shape[layer_idx]["shape"]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "Tanh":
                self.MLP.append(nn.Tanh())
                data_shape = {"channel": self.data_shape[layer_idx]["channel"],
                              "shape"  : self.data_shape[layer_idx]["shape"]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "Maxpool":
                self.MLP.append(nn.MaxPool3d(kernel_size=layer["kernel_size"],
                                             stride=layer["stride"],
                                             padding=0,
                                             return_indices=True))
                
                data_shape = {"channel": self.data_shape[layer_idx]["channel"],
                              "shape"  : [convShape(e, layer["kernel_size"], layer["stride"], layer["padding"]) 
                                          for e in self.data_shape[layer_idx]["shape"]]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "LocalRespNorm":
                self.MLP.append(nn.LocalResponseNorm(size=layer["size"]))
                data_shape = {"channel": self.data_shape[layer_idx]["channel"],
                              "shape"  : self.data_shape[layer_idx]["shape"]}
                self.data_shape.append(data_shape)
            
            elif layer["type"] == "Flatten":
                assert layer["in_features"] == [self.data_shape[layer_idx]["channel"]] + self.data_shape[layer_idx]["shape"]
                data_shape = {"channel": 0,
                              "shape": int(np.prod(layer["in_features"]))}
                self.data_shape.append(data_shape)
            
            elif layer["type"] == "Linear":
                assert layer["in_features"] == self.data_shape[layer_idx]["shape"]
                self.MLP.append(nn.Linear(in_features=layer["in_features"],
                                                                          out_features=layer["out_features"]))

                data_shape = {"channel": 0,
                              "shape"  : layer["out_features"]}
                self.data_shape.append(data_shape)

        self.linear_means = nn.Linear(self.data_shape[-1]["shape"], latent_size)
        self.linear_log_var = nn.Linear(self.data_shape[-1]["shape"], latent_size)

        data_shape = {}
        data_shape["channel"] = 0
        data_shape["shape"] = latent_size
        self.data_shape.append(data_shape)

    def forward(self, x, c=None):
        pool_idx_set = []
        layer_idx = 0
        for layer in (self.encoder_layer):
            if layer["type"] == "Maxpool":
                x, pool_idx = self.MLP[layer_idx](x)
                pool_idx_set.append(pool_idx)
            elif layer["type"] == "Flatten":
                x = x.view(x.size(0), -1)
                layer_idx = layer_idx - 1
            else:
                x = self.MLP[layer_idx](x)
            layer_idx = layer_idx + 1

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars, pool_idx_set


class Decoder(nn.Module):
    def __init__(self, decoder_layer, latent_size):
        nn.Module.__init__(self)
        self.decoder_layer = decoder_layer
        self.data_shape = []
        data_shape = {"channel": 0,
                      "shape"  : latent_size}
        self.data_shape.append(data_shape)

        assert self.decoder_layer[0]["type"] == "Linear" or "Relu"
        if self.decoder_layer[0]["type"] == "Linear":
            self.linear = nn.Linear(latent_size, self.decoder_layer[0]["in_features"])
            data_shape = {"channel": 0,
                      "shape"  : self.decoder_layer[0]["in_features"]}
        elif self.decoder_layer[0]["type"] == "Relu":
            assert self.decoder_layer[1]["type"] == "Linear" or "Relu"
            self.linear = nn.Linear(latent_size, self.decoder_layer[1]["in_features"])
            data_shape = {"channel": 0,
                      "shape"  : self.decoder_layer[1]["in_features"]}
        self.data_shape.append(data_shape)

        self.MLP = nn.ModuleList()

        for layer_idx, layer in enumerate(self.decoder_layer):
            if layer["type"] == "Transconv":
                assert self.data_shape[layer_idx+1]["channel"] == layer["in_channel"]
                self.MLP.append(nn.ConvTranspose3d(in_channels=layer["in_channel"],
                                                   out_channels=layer["out_channel"],
                                                   kernel_size=layer["kernel_size"],
                                                   stride=layer["stride"],
                                                   padding=0,
                                                   bias=True))

                data_shape = {"channel": layer["out_channel"],
                              "shape"  : [convTransShape(e, layer["kernel_size"], layer["stride"], layer["padding"]) 
                                         for e in self.data_shape[layer_idx+1]["shape"]]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "Relu":
                self.MLP.append(nn.ReLU())
                data_shape = {"channel": self.data_shape[layer_idx+1]["channel"],
                              "shape"  : self.data_shape[layer_idx+1]["shape"]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "Tanh":
                self.MLP.append(nn.Tanh())
                data_shape = {"channel": self.data_shape[layer_idx+1]["channel"],
                              "shape"  : self.data_shape[layer_idx+1]["shape"]}
                self.data_shape.append(data_shape)

            elif layer["type"] == "Maxunpool":
                self.MLP.append(nn.MaxUnpool3d(kernel_size=layer["kernel_size"],
                                               stride=layer["stride"],
                                               padding=0))
                data_shape = {"channel": self.data_shape[layer_idx+1]["channel"],
                              "shape"  : [convTransShape(e, layer["kernel_size"], layer["stride"], layer["padding"]) 
                                          for e in self.data_shape[layer_idx+1]["shape"]]}
                self.data_shape.append(data_shape)
            
            elif layer["type"] == "Unflatten":
                assert self.data_shape[layer_idx+1]["shape"] == np.prod(layer["out_features"])
                data_shape = {"channel": layer["out_features"][0],
                              "shape"  : layer["out_features"][1:]}
                self.data_shape.append(data_shape)
            
            elif layer["type"] == "Linear":
                assert self.data_shape[layer_idx+1]["shape"] == layer["in_features"]
                self.MLP.append(nn.Linear(in_features=layer["in_features"],
                                          out_features=layer["out_features"]))
                data_shape = {"channel": 0,
                              "shape"  : layer["out_features"]}
                self.data_shape.append(data_shape)

        self.sigmoid=nn.Sigmoid()
        

    def forward(self, z, pool_idx_set, c=None):
        x = self.linear(z)
        layer_idx = 0
        for layer in (self.decoder_layer):
            if layer["type"] == "Maxunpool":
                pool_idx = pool_idx_set.pop()
                x = self.MLP[layer_idx](x,pool_idx)
            elif layer["type"] == "Unflatten":
                x = x.view(x.size(0), layer["out_features"][0], layer["out_features"][1], layer["out_features"][2], layer["out_features"][3])
                layer_idx = layer_idx - 1
            else:
                x = self.MLP[layer_idx](x)
            layer_idx = layer_idx + 1
        x = self.sigmoid(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_data_shape, encoder_layer, decoder_layer, latent_size, device):
        nn.Module.__init__(self)
        self.device = device
        
        assert type(input_data_shape) == dict
        assert type(encoder_layer) == list
        assert type(latent_size) == int
        assert type(decoder_layer) == list

        self.latent_size = latent_size

        self.encoder = Encoder(input_data_shape, encoder_layer, latent_size)
        self.decoder = Decoder(decoder_layer, latent_size)

    def forward(self, x, c=None):
        # if x.dim() > 2:
        #     x = x.view(-1, 28*28)
        batch_size = x.size(0)

        means, log_var, pool_idx_set = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=self.device)
        z = eps * std + means

        recon_x = self.decoder(z, pool_idx_set, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None, z=None):

        batch_size = n
        if z is None:
            z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x

    def forward_once(self, x):
        means, log_var = self.encoder(x, None)

        std = torch.exp(0.5 * log_var)
        # eps = torch.randn([batch_size, self.latent_size])
        z = means

        recon_x = self.decoder(z, None)

        return recon_x, means, log_var, z

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
            if i+2 < len(layer_sizes):
                self.MLP.add_module(
                    name="D{:d}".format(i), module=nn.Dropout(0.5))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())


        self.MLP.apply(init_weights)

    def forward(self, x):
        x = self.MLP(x)
        return x

class CollNet(nn.Module):
    def __init__(self,input_data_shape, encoder_layer, decoder_layer, fc_layer_sizes, batch_size, latent_size, device):
        
        nn.Module.__init__(self)

        assert type(input_data_shape) == dict
        assert type(encoder_layer) == list
        assert type(latent_size) == int
        assert type(decoder_layer) == list

        self.latent_size = latent_size

        self.local_occuapncy_vae = VAE(input_data_shape, encoder_layer, decoder_layer, latent_size, device)
        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size)

    def forward(self, x_q, x_voxel):
        """
        x: input nerf_q(21), voxel(4096:16x16x16)
        """
        recon_x, mean, log_var, z = self.local_occuapncy_vae(x_voxel)
        x_fc = torch.cat([x_q,z], dim=1)
        y = self.fc(x_fc)
        
        return y, x_voxel, recon_x, mean, log_var
        

