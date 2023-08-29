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
import shutil

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm

NUM_QSET = 10000
NUM_ENVSET = 100
NUM_LINK = 9

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
        print('coll_shape', self.coll.shape)

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

        return np.array(self.grid[idx_tmp],dtype=np.float32), np.array(self.nerf_q[idx],dtype=np.float32), np.array(self.coll[idx],dtype=np.int32)

def main(args):
    vae_latent_size = args.vae_latent_size
    train_ratio = 0.999
    test_ratio = 1 - train_ratio
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/grid/' + data_dir
    chkpt_dir = 'model/checkpoints/grid/' + data_dir
    model_dir = 'model/grid/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    folder_path = 'model/checkpoints/grid/'
    num_save = 3
    order_list = sorted(os.listdir(folder_path), reverse=True)
    remove_folder_list = order_list[num_save:]
    for rm_folder in remove_folder_list:
        shutil.rmtree('log/grid/'+rm_folder)
        shutil.rmtree('model/checkpoints/grid/'+rm_folder)
        shutil.rmtree('model/grid/'+rm_folder)
    
    suffix = 'lat{}_rnd{}'.format(vae_latent_size, args.seed)

    file_name = "dataset/2023_08_26_05_06_37/box_grid.pickle"
    # 2023_08_23_23_57_09 : 32
    # 2023_08_24_00_43_57 : 16
    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    """
    layer size = [21+len(z), hidden1, hidden2, num_links(possibility to collide)]
    """
    layer_size = [21+vae_latent_size, 256, 256, NUM_LINK]

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
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
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
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(
            recon_x.view(-1, n_grid, n_grid, n_grid), x.view(-1, n_grid, n_grid, n_grid), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)


    def loss_fn_fc(coll_hat, coll):
        weights = [1, 2, 3, 4, 5, 6, 7, 8, 8]
        weights = [i / sum(weights) for i in weights]
        weights = torch.Tensor(weights).to(device)
        BCE = torch.nn.BCEWithLogitsLoss(reduction="mean", weight=weights)
        loss = BCE(coll_hat, coll)
        return loss
    
    NN_param = {}
    NN_param["input_data_shape"] = {"channel": 1, "shape":[n_grid, n_grid, n_grid]}

    # for num_grid = 16 
    # NN_param["encoder_layer"] = [{"type": "Conv",    "in_channel": 1, "out_channel": 4, "kernel_size": 3, "stride": 1},
    #                              {"type": "Relu"},
    #                             #  {"type": "Maxpool",                                    "kernel_size": 2, "stride": 2},
    #                              {"type": "Conv",    "in_channel": 4, "out_channel": 8, "kernel_size": 2, "stride": 2},
    #                              {"type": "Relu"},
    #                             #  {"type": "Maxpool",                                    "kernel_size": 2, "stride": 2},
    #                              {"type": "Flatten", "in_features":[8,7,7,7]},
    #                              {"type": "Linear", "in_features": int(8*7*7*7), "out_features": 256},
    #                              {"type": "Relu"}]
    # NN_param["decoder_layer"] = [{"type": "Relu"},
    #                              {"type": "Linear", "in_features": 256,          "out_features": int(8*7*7*7)},
    #                              {"type": "Unflatten", "out_features": [8,7,7,7]},
    #                             #  {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2},
    #                              {"type": "Relu"},
    #                              {"type": "Transconv", "in_channel": 8, "out_channel": 4, "kernel_size": 2, "stride": 1},
    #                             #  {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2},
    #                              {"type": "Relu"},
    #                              {"type": "Transconv", "in_channel": 4, "out_channel": 1, "kernel_size": 3, "stride": 1}]
    
    # for num_grid = 32
    NN_param["encoder_layer"] = [{"type": "Conv",    "in_channel": 1, "out_channel": 8, "kernel_size": 3, "stride": 1, "padding": 0},
                                 {"type": "Relu"},
                                 {"type": "Maxpool",                                     "kernel_size": 2, "stride": 2, "padding": 0},
                                 {"type": "LocalRespNorm", "size":2},
                                 {"type": "Conv",    "in_channel": 8, "out_channel": 16, "kernel_size": 3, "stride": 2, "padding": 0},
                                 {"type": "Relu"},
                                #  {"type": "Maxpool",                                    "kernel_size": 2, "stride": 2, "padding": 0},
                                 {"type": "Flatten", "in_features":[16,7,7,7]},
                                 {"type": "Linear", "in_features": int(16*7*7*7), "out_features": 1024},
                                 {"type": "Relu"}]
    NN_param["decoder_layer"] = [{"type": "Relu"},
                                 {"type": "Linear", "in_features": 1024,          "out_features": int(16*7*7*7)},
                                 {"type": "Unflatten", "out_features": [16,7,7,7]},
                                #  {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2, "padding": 0},
                                 {"type": "Relu"},
                                 {"type": "Transconv", "in_channel": 16, "out_channel": 4, "kernel_size": 3, "stride": 2, "padding": 0}, 
                                #  {"type": "Maxunpool",                                    "kernel_size": 2, "stride": 2, "padding": 0},
                                 {"type": "Relu"},
                                 {"type": "Transconv", "in_channel": 4, "out_channel": 1, "kernel_size": 4, "stride": 2, "padding": 0}]
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

    # optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(collnet.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                        threshold=0.001, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0




    for grid, nerf_q, coll in test_data_loader:
        test_nerf_q, test_grid, test_coll = nerf_q.to(device).squeeze(), grid.to(device), coll.type(torch.float32).to(device).squeeze()

    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)

        for grid, nerf_q, coll in loader_tqdm:
            nerf_q, grid, coll = nerf_q.to(device).squeeze(), grid.to(device), coll.type(torch.float32).to(device).squeeze()
            
            collnet.train()
            loss_vae = []
            loss_fc = []
            loss = []
        
            with torch.cuda.amp.autocast():
                coll_hat, x_voxel, recon_x, mean, log_var = collnet.forward(nerf_q, grid)
                # coll_hat = coll_hat.view(-1, NUM_LINK)

                train_loss_vae = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid**3
                train_loss_fc = loss_fn_fc(coll_hat, coll)
                train_loss = train_loss_vae + train_loss_fc

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_vae.append(train_loss_vae.item())
            loss_fc.append(train_loss_fc.item())
            loss.append(train_loss.item())

            coll_hat_bin = (coll_hat > 0.5).type(torch.int32)
            coll_bin = coll.type(torch.int32)

            train_accuracy = []
            for link in range(NUM_LINK):
                train_accuracy.append( (coll_bin == coll_hat_bin).sum(dim=0)[link].item() / coll.size(dim=0) )

            # train_accuracy_all = sum(train_accuracy) / NUM_LINK


            
        collnet.eval()
        with torch.cuda.amp.autocast() and torch.no_grad():
            coll_hat, x_voxel, recon_x, mean, log_var = collnet.forward(test_nerf_q, test_grid)

            loss_vae_test = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid**3
            loss_fc_test = loss_fn_fc(coll_hat, test_coll)
            loss_test = loss_vae_test + loss_fc_test

        coll_hat_bin = (coll_hat > 0.5).type(torch.int32)
        coll_bin = test_coll.type(torch.int32)

        test_accuracy = []
        accuracy = []

        for link in range(NUM_LINK):
            test_accuracy.append( (coll_bin == coll_hat_bin).sum(dim=0)[link].item() / test_coll.size(dim=0) )

            truth_positives = (coll_bin == 1).sum(dim=0)[link].item() 
            truth_negatives = (coll_bin == 0).sum(dim=0)[link].item() 

            confusion_vector = (coll_hat_bin / coll_bin)[:, link]
            
            true_positives = torch.sum(confusion_vector == 1).item()
            false_positives = torch.sum(confusion_vector == float('inf')).item()
            true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives = torch.sum(confusion_vector == 0).item()

            accuracy.append({'tp': true_positives / truth_positives,
                             'fp': false_positives / truth_negatives,
                             'tn': true_negatives / truth_negatives,
                             'fn': false_negatives / truth_positives})

        if epoch == 0:
            min_loss = loss_test

        scheduler.step(loss_test)
        loss_vae_train = np.mean(loss_vae)
        loss_fc_train = np.mean(loss_fc)
        loss_train = np.mean(loss)

        if loss_test < min_loss:
            e_notsaved = 0
            print('saving model', loss_test.item())
            checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{:.4f}_{}_grid'.format(loss_test.item(), model_name, epoch, vae_latent_size, args.seed) + '.pkl'
            torch.save(collnet.state_dict(), checkpoint_model_name)
            min_loss = loss_test
        print("Epoch: {} (Saved at {})".format(epoch, epoch-e_notsaved))
        print("[Train] vae loss: {:.3f} fc loss: {:.3f} total loss: {:.3f}".format(loss_vae_train.item(),loss_fc_train.item(),loss_train.item()))
        print("[Test]  vae loss: {:.3f} fc loss: {:.3f} total loss: {:.3f}".format(loss_vae_test.item(),loss_fc_test.item(),loss_test.item()))
        print("[Train] Accuracy: {}".format(train_accuracy))
        print("[Test]  Accuracy: {}".format(test_accuracy))
        print("coll    : {}".format(coll_bin[0]))
        print("coll_hat: {}".format(coll_hat[0]))
        print("=========================================================================================")

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train Loss(VAE, FC, Total): {}, {}, {} / Test Loss(VAE, FC, Total): {}, {}, {} / Train Accuracy: {} / Test Accuracy: {} / TP: {} / FP: {} / TN: {} / FN: {}".format(epoch,
                                                                                                                                                                                                                   epoch - e_notsaved,
                                                                                                                                                                                                                   loss_vae_train,
                                                                                                                                                                                                                   loss_fc_train,
                                                                                                                                                                                                                   loss_train,
                                                                                                                                                                                                                   loss_vae_test,
                                                                                                                                                                                                                   loss_fc_test,
                                                                                                                                                                                                                   loss_test,
                                                                                                                                                                                                                   train_accuracy,
                                                                                                                                                                                                                   test_accuracy,
                                                                                                                                                                                                                   [link_acc["tp"] for link_acc in accuracy],
                                                                                                                                                                                                                   [link_acc["fp"] for link_acc in accuracy],
                                                                                                                                                                                                                   [link_acc["tn"] for link_acc in accuracy],
                                                                                                                                                                                                                   [link_acc["fn"] for link_acc in accuracy]))
            
        e_notsaved += 1
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