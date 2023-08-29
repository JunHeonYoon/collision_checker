import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_size):
        nn.Module.__init__(self)

        self.conv_MLP1 = nn.Sequential()
        self.conv_MLP1.add_module(name="C0", module=nn.Conv2d(in_channels=4,
                                                              out_channels=8,
                                                              kernel_size=8,
                                                              stride=3,
                                                              padding=0,
                                                              bias=True))
        self.conv_MLP1.add_module(name="A0", module=nn.ReLU())
        self.conv_MLP1.add_module(name="P0", module=nn.MaxPool2d(kernel_size=3,
                                                                 stride=2,
                                                                 return_indices=True))
        self.conv_MLP2 = nn.Sequential()
        self.conv_MLP2.add_module(name="C1", module=nn.Conv2d(in_channels=8,
                                                              out_channels=16,
                                                              kernel_size=5,
                                                              stride=3,
                                                              padding=0,
                                                              bias=True))
        self.conv_MLP2.add_module(name="A1", module=nn.ReLU())
        self.conv_MLP2.add_module(name="P1", module=nn.MaxPool2d(kernel_size=2,
                                                                 stride=2,
                                                                 padding=0,
                                                                 return_indices=True))
        self.conv_MLP3 = nn.Sequential()
        self.conv_MLP3.add_module(name="C2", module=nn.Conv2d(in_channels=16,
                                                              out_channels=32,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=0,
                                                              bias=True))
        self.conv_MLP3.add_module(name="A2", module=nn.ReLU())

        self.fc_MLP = nn.Sequential()
        self.fc_MLP.add_module(name="L2", module=nn.Linear(in_features=32*6*6,
                                                           out_features=128))
        self.fc_MLP.add_module(name="A2", module=nn.ReLU())


        self.linear_means = nn.Linear(128, latent_size)
        self.linear_log_var = nn.Linear(128, latent_size)

    def forward(self, x, c=None):
        x, pool_idx1 = self.conv_MLP1(x)
        x, pool_idx2 = self.conv_MLP2(x)
        if x.dim() < 4:
            x = x.view(1, 32, 6, 6)
        x = x.view(x.size(0), -1)
        x = self.fc_MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars, pool_idx1, pool_idx2


class Decoder(nn.Module):
    def __init__(self, latent_size):
        nn.Module.__init__(self)

        self.fc_MLP = nn.Sequential()
        self.fc_MLP.add_module(name="L0", module=nn.Linear(in_features=latent_size,
                                                           out_features=128))
        self.fc_MLP.add_module(name="A0", module=nn.ReLU())
        self.fc_MLP.add_module(name="L1", module=nn.Linear(in_features=128,
                                                           out_features=32*6*6))
        
        self.conv_MLP1 = nn.Sequential()
        self.conv_MLP1.add_module(name="A2", module=nn.ReLU())
        self.conv_MLP1.add_module(name="C2", module=nn.ConvTranspose2d(in_channels=32,
                                                                       out_channels=16,
                                                                       kernel_size=3,
                                                                       stride=2,
                                                                       padding=0,
                                                                       bias=True))
        self.conv_P1=nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.conv_MLP2 = nn.Sequential()
        self.conv_MLP2.add_module(name="A3", module=nn.ReLU())
        self.conv_MLP2.add_module(name="C3", module=nn.ConvTranspose2d(in_channels=16,
                                                                       out_channels=8,
                                                                       kernel_size=5,
                                                                       stride=3,
                                                                       padding=0,
                                                                       bias=True))

        self.conv_P2 =nn.MaxUnpool2d(kernel_size=3, stride=2, padding=0)
        self.conv_MLP2 = nn.Sequential()
        self.conv_MLP2.add_module(name="A4", module=nn.ReLU())
        self.conv_MLP2.add_module(name="C4", module=nn.ConvTranspose2d(in_channels=8,
                                                                       out_channels=4,
                                                                       kernel_size=8,
                                                                       stride=3,
                                                                       padding=0,
                                                                       bias=True))        

    def forward(self, z, pool_idx1, pool_idx2, c=None):
        x = self.fc_MLP(z)
        x = x.view(x.size(0), 32, 6, 6)
        if pool_idx1.dim() < 4:
            pool_idx1 = pool_idx1.unsqueeze(0)
            pool_idx2 = pool_idx2.unsqueeze(0)
        x = self.conv_P1(x, pool_idx2)
        x = self.conv_MLP1(x)
        x = self.conv_P2(x, pool_idx1)
        x = self.conv_MLP2(x)

        return x

class VAE(nn.Module):
    def __init__(self, latent_size, device):
        nn.Module.__init__(self)
        self.device = device

        # assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        # assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x, c=None):
        # if x.dim() > 2:
        #     x = x.view(-1, 28*28)
        batch_size = x.size(0)

        means, log_var, pool_idx1, pool_idx2 = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=self.device)
        z = eps * std + means

        recon_x = self.decoder(z, pool_idx1, pool_idx2, c)

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

        # assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        # assert type(decoder_layer_sizes) == list

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
        

