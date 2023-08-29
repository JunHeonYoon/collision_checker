from __future__ import division
import os
import time
import torch
import argparse
from collections import defaultdict
import pickle
import numpy as np
from models_grid import CollNet
import datetime as dt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NUM_QSET = 5000
NUM_ENVSET = 500

class CollisionNetDataset(Dataset):
    """
    data pickle contains dict
        'grid_len'  : the length of grid voxel
        'grid_size' : the size of grid voxel
        'grid'      : occupancy data
        'nerf_q'    : nerf joint state data [q, cos(q), sin(q)]
        'coll'      : collision vector data
    """
    def __init__(self, file_name,):
        def data_load():
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
            self.grid_size = dataset['grid_size']
            self.grid_len = dataset['grid_len']
            return dataset['nerf_q'], dataset['grid'], dataset['coll']
        self.nerf_q, self.grid, self.coll = data_load()
        print ('grid shape', self.grid.shape)

    def __len__(self):
        return len(self.nerf_q)

    def get_grid_len(self):
        return self.grid_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        idx_tmp = [idx_tmp // NUM_QSET for idx_tmp in idx]
        # idx_tmp = int(idx // NUM_QSET)

        return np.array(self.grid[idx_tmp],dtype=np.float32), np.array(self.nerf_q[idx],dtype=np.float32), np.array(self.coll[idx],dtype=np.float32)

def main(args):
    vae_latent_size = args.vae_latent_size
    
    # link_num = args.link_num

    date = dt.datetime.now()
    data_dir = "{}_{}_{}_{}_{}_{}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/grid/' + data_dir
    chkpt_dir = 'model/checkpoints/grid/' + data_dir
    model_dir = 'model/grid/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    suffix = 'lat{}_rnd{}'.format(vae_latent_size, args.seed)

    file_name = "dataset/box_grid_16.pickle"
    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    """
    layer size = [21+len(z), hidden1, hidden2, 2(free / collide)]
    """
    layer_size = [21+vae_latent_size, 255, 255, 2]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    ts = time.time()

    print('loading data ...')
    read_time = time.time()
    dataset = CollisionNetDataset(
        file_name=file_name)
    n_grid = dataset.get_grid_len()
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # train_dataset = torch.utils.data.Subset(dataset, range(0,train_size))
    # test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size+test_size))
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    
    def loss_fn_vae(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, n_grid, n_grid, n_grid), x.view(-1, n_grid, n_grid, n_grid), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    def loss_fn_fc(y_hat, y):
        CE = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = CE(y_hat, y)
        return loss
    
    NN_param = {}
    NN_param["input_data_shape"] = {"channel": 1, "shape":[n_grid, n_grid, n_grid]}
    NN_param["encoder_layer"] = [{"type": "Conv",    "in_channel": 1, "out_channel": 4, "kernel_size": 3, "stride": 1},
                                 {"type": "Relu"},
                                 {"type": "Maxpool",                                    "kernel_size": 2, "stride": 2},
                                 {"type": "Conv",    "in_channel": 4, "out_channel": 8, "kernel_size": 2, "stride": 1},
                                 {"type": "Relu"},
                                 {"type": "Maxpool",                                    "kernel_size": 2, "stride": 2},
                                 {"type": "Flatten", "in_features":[8,3,3,3]},
                                 {"type": "Linear", "in_features": int(8*3*3*3), "out_features": 128},
                                 {"type": "Relu"}]
    NN_param["decoder_layer"] = [{"type": "Relu"},
                                 {"type": "Linear", "in_features": 128,          "out_features": int(8*3*3*3)},
                                 {"type": "Unflatten", "out_features": [8,3,3,3]},
                                 {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2},
                                 {"type": "Relu"},
                                 {"type": "Transconv", "in_channel": 8, "out_channel": 4, "kernel_size": 2, "stride": 1},
                                 {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2},
                                 {"type": "Relu"},
                                 {"type": "Transconv", "in_channel": 4, "out_channel": 1, "kernel_size": 3, "stride": 1}]
    NN_param["fc_layer"] = layer_size
    NN_param["latent_size"] = vae_latent_size

    collnet = CollNet(input_data_shape=NN_param["input_data_shape"] ,
                      encoder_layer=NN_param["encoder_layer"],
                      decoder_layer=NN_param["decoder_layer"],
                      fc_layer_sizes=NN_param["fc_layer"],
                      latent_size=NN_param["latent_size"],
                      batch_size=args.batch_size,
                      device=device).to(device)
    print(collnet)
    with open(model_dir + "model_param.pickle", "wb") as f:
        pickle.dump(NN_param,f)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    logs = defaultdict(list)
    # clear log
    with open(log_file_name, 'w'):
        pass
    min_loss = 1e100
    for iteration, (grid, nerf_q, coll) in enumerate(test_data_loader):
        test_nerf_q, test_grid, test_coll = nerf_q.to(device).squeeze(), grid.to(device), coll.type(torch.LongTensor).to(device).squeeze()
    for epoch in range(args.epochs):
        collnet.train()

        for iteration, (grid, nerf_q, coll) in enumerate(train_data_loader):
            nerf_q, grid, coll = nerf_q.to(device).squeeze(), grid.to(device), coll.type(torch.LongTensor).to(device).squeeze()
            
            x_q = nerf_q
            x_g = grid
            y = coll

            y_hat, x_voxel, recon_x, mean, log_var = collnet(x_q, x_g)

            loss_vae = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid**3
            loss_fc = loss_fn_fc(y_hat.view(-1,2), y) # sum -> mean, divison remove

            loss = loss_vae + loss_fc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_data_loader)-1:
                print("======================================================================")
                print("[Train] Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_data_loader)-1, loss.item()))

                _, y_hat_bin = torch.max(y_hat, dim=1)
                y_bin = y

                train_accuracy = (y_bin == y_hat_bin).sum().item()
                
                with torch.no_grad():
                    # x = torch.cat([test_nerf_q,test_grid], dim=1)
                    x_q = test_nerf_q
                    x_g = test_grid
                    y = test_coll
                    collnet.eval()
                    y_hat, x_voxel, recon_x, mean, log_var = collnet(x_q, x_g)

                    loss_vae_test = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid**3
                    loss_fc_test = loss_fn_fc(y_hat.view(-1,2), y)

                    loss_test = loss_vae_test + loss_fc_test
                    
                    _, y_hat_bin = torch.max(y_hat, dim=1)
                    y_bin = (y)

                    truth_positives = (y_bin == 1).sum().item() 
                    truth_negatives = (y_bin == 0).sum().item() 

                    confusion_vector = y_hat_bin / y_bin
                    
                    true_positives = torch.sum(confusion_vector == 1).item()
                    false_positives = torch.sum(confusion_vector == float('inf')).item()
                    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
                    false_negatives = torch.sum(confusion_vector == 0).item()

                    test_accuracy = (y_bin == y_hat_bin).sum().item()
                    accuracy = {}
                    accuracy['tp'] = true_positives / truth_positives
                    accuracy['fp'] = false_positives / truth_negatives
                    accuracy['tn'] = true_negatives / truth_negatives
                    accuracy['fn'] = false_negatives / truth_positives

                lv = loss_vae_test.item()
                lf = loss_fc_test.item()
                lt = loss_test.item()

                lv0 = loss_vae.item()
                lf0 = loss_fc.item()
                lt0 = loss.item()
                train_accuracy = float(train_accuracy)/coll.size(dim=0)
                test_accuracy = float(test_accuracy)/test_nerf_q.size(dim=0)
                print("[Test] vae loss: {:.3f} fc loss: {:.3f} total loss: {:.3f}".format(lv,lf,lt))
                print("[Test] Accuracy: Train: {} / Test: {}".format(train_accuracy, test_accuracy))
                print("y    : {}".format(y_bin[0:10]))
                print("y_hat: {}".format(y_hat_bin[0:10]))
                print("======================================================================")
                    
                if lt < min_loss:
                    min_loss = loss.item()
                    
                    checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{:04d}_{:.4f}_{}_grid'.format(lt, model_name, epoch, iteration, vae_latent_size, args.seed) + '.pkl'
                    torch.save(collnet.state_dict(), checkpoint_model_name)

                if iteration == 0:
                    with open(log_file_name, 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, lv,lf,lt,test_accuracy,lv0,lf0,lt0,train_accuracy,accuracy['tp'],accuracy['fp'],accuracy['tn'],accuracy['fn']))
    torch.save(collnet.state_dict(), model_dir+'ss{}.pkl'.format(model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--vae_latent_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=100)

    args = parser.parse_args()
    main(args)