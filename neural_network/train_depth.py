from __future__ import division
import os
import time
import torch
import argparse
from collections import defaultdict
import pickle
import numpy as np
from models_depth import CollNet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NUM_QSET = 5000
NUM_ENVSET = 500
NUM_CAM = 4

class CollisionNetDataset(Dataset):
    """
    data pickle contains dict
        'depth_len'  : the length of depth data
        'depth_size' : the size of depth data
        'depth'      : occupancy data
        'nerf_q'    : nerf joint state data [q, cos(q), sin(q)]
        'coll'      : collision vector data
    """
    def __init__(self, file_name,):
        def data_load():
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
            self.depth_size = dataset['depth_size']
            self.depth_len = dataset['depth_len']
            return dataset['nerf_q'], dataset['depth'], dataset['coll']
        self.nerf_q, self.depth, self.coll = data_load()
        print ('depth shape', self.depth.shape)

    def __len__(self):
        return len(self.nerf_q)

    def get_depth_len(self):
        return self.depth_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        idx_tmp = [idx_tmp // NUM_QSET for idx_tmp in idx]
        return np.array(self.depth[idx_tmp],dtype=np.float32), np.array(self.nerf_q[idx],dtype=np.float32), np.array(self.coll[idx],dtype=np.float32)

def main(args):
    vae_latent_size = args.vae_latent_size
    
    link_num = args.link_num

    # directory = args.dataset
    log_dir = 'log/' +'/depth/'
    chkpt_dir = 'model/checkpoints/' + '/depth/'
    model_dir = 'model/'+ '/depth/'

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    suffix = 'lat{}_rnd{}'.format(vae_latent_size, args.seed)

    file_name = "dataset/box_depth.pickle"
    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    """
    layer size = [21+len(z), hidden1, hidden2, link_num]
    """
    layer_size = [21+vae_latent_size, 255, 255, 255] + [link_num]

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
    n_depth = dataset.get_depth_len()
    train_size = int(0.999 * len(dataset))
    test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = torch.utils.data.Subset(dataset, range(0,train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size+test_size))
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    
    def loss_fn_vae(recon_x, x, mean, log_var):
        # BCE = torch.nn.functional.binary_cross_entropy(
        #     recon_x.view(-1, n_depth, n_depth, n_depth), x.view(-1, n_depth, n_depth, n_depth), reduction='sum')
        mse_fn = torch.nn.MSELoss(reduction='sum')
        BCE = mse_fn(recon_x.view(-1, NUM_CAM, n_depth, n_depth), x.view(-1, NUM_CAM, n_depth, n_depth))
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    def loss_fn_fc(y_hat, y):
        CE = torch.nn.CrossEntropyLoss(reduce="mean")
        loss = CE(y_hat, y)
        return loss

    collnet = CollNet(
        latent_size=vae_latent_size,
        fc_layer_sizes=layer_size,
        batch_size=args.batch_size,
        device=device).to(device)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    logs = defaultdict(list)
    # clear log
    with open(log_file_name, 'w'):
        pass
    min_loss = 1e100
    for iteration, (depth, nerf_q, coll) in enumerate(test_data_loader):
        test_nerf_q, test_depth, test_coll = nerf_q.to(device).squeeze(), depth.to(device), coll.type(torch.LongTensor).to(device).squeeze()
    for epoch in range(args.epochs):
        collnet.train()

        for iteration, (depth, nerf_q, coll) in enumerate(train_data_loader):
            nerf_q, depth, coll = nerf_q.to(device).squeeze(), depth.to(device), coll.type(torch.LongTensor).to(device).squeeze()
            
            x_q = nerf_q
            x_g = depth
            y = coll

            y_hat, x_depth, recon_x, mean, log_var = collnet(x_q, x_g)

            loss_vae = loss_fn_vae(recon_x, x_depth, mean, log_var) / (n_depth**2*NUM_CAM)
            loss_fc = loss_fn_fc(y_hat.view(-1,link_num), y) # sum -> mean, divison remove

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
                    x_q = test_nerf_q
                    x_g = test_depth
                    y = test_coll
                    collnet.eval()
                    y_hat, x_depth, recon_x, mean, log_var = collnet(x_q, x_g)

                    loss_vae_test = loss_fn_vae(recon_x, x_depth, mean, log_var) / (n_depth**2*NUM_CAM)
                    loss_fc_test = loss_fn_fc(y_hat.view(-1,link_num), y)

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
                    
                    checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{:04d}_{:.4f}_{}_depth'.format(lt, model_name, epoch, iteration, vae_latent_size, args.seed) + '.pkl'
                    torch.save(collnet.state_dict(), checkpoint_model_name)

                if iteration == 0:
                    with open(log_file_name, 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, lv,lf,lt,test_accuracy,lv0,lf0,lt0,train_accuracy,accuracy['tp'],accuracy['fp'],accuracy['tn'],accuracy['fn']))
    torch.save(collnet.state_dict(), model_dir+'ss{}.pkl'.format(model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--vae_latent_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--link_num", type=int, default=2)
    # parser.add_argument("--log_file_name", type=str, default="box")
    # parser.add_argument("--pose_noise", type=float, default=0.00)
    # parser.add_argument("--dataset", type=str, default='box')

    args = parser.parse_args()
    main(args)