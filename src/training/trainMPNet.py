from architechtures.AE_model import AENet3
from architechtures.agent_model import MPNet1
from architechtures.selfsupervised_model import SelfSupervisedModel
from data_generation.dataset import MPNetDataset, get_train_test_val
from utilities.utils import ChamferDistance

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

from tqdm import tqdm, trange
import os
import copy
import time
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


###################################################################
###### Train the MPNet model
###################################################################
# loss_keys = ['loss', 'loss_collided', 'loss_collision_pos', 'loss_robot_links', 'pt_clouds']
loss_keys = ['loss', 'loss_action']

def lossCriterion(pred, target):
    """the loss fucnction for MPNet
    """
    loss = dict.fromkeys(loss_keys, None)
    loss_val = 0

    loss['loss_action'] = F.l1_loss(pred['action'], target['action'])
    loss_val += loss['loss_action']
    
    loss['loss'] = loss_val
    return loss

class lossCriterionAE():
    def __init__(self) -> None:
        self.recons_lossFn = ChamferDistance()

    def __call__(self, pred, target):
        return {'loss': self.recons_lossFn(pred, target)}
    
class lossCriterionSelfSupervised():
    def __init__(self) -> None:
        self.recons_lossFn = ChamferDistance()

    def __call__(self, pred, target):
        """Computes the self-supervised loss
        Args:
            pred (dict): dictionary containing the NN predicitions
            target (dict): dictionary containing the labels

        Returns:
            dict: loss dictionary 
        """
        loss = dict.fromkeys(loss_keys, None)
        loss_weights = dict.fromkeys(loss_keys, 1.0)
 
        loss['loss_collided'] = F.binary_cross_entropy(pred['collided'], target['collided'])
        loss['loss_collision_pos'] = F.l1_loss(pred['collision_pos'], target['collision_pos'])
        loss['loss_robot_links'] = F.l1_loss(pred['robot_links'], target['robot_links'])
        loss['pt_clouds'] = self.recons_lossFn(pred['pt_clouds'], target['pt_clouds'])
        
        loss_val = 0
        for k in loss.keys():
            loss_val += loss_weights[k] * loss[k]
        loss['loss'] = loss_val
        return loss 

class TrainerMPNet(object):
    """Training class for MPNet or obstacle Auto-Encoder
    """
    def __init__(self, is_MPNET) -> None:
        self.is_MPNET = is_MPNET
        self.num_epochs = 1000
        self.directory = './logs/' + datetime.datetime.now().strftime("%m%d_%H_%M/")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        ### loss function
        self.criterion = lossCriterion if self.is_MPNET else lossCriterionAE()

        ### data loader
        self.loader_train, self.loader_test, self.loader_val = get_train_test_val(self.is_MPNET)
        print('data loaded')
        ### network and optimizer
        dim = 7
        obs_dim = 128
        self.net = MPNet1(state_dim=obs_dim + 2*dim, action_dim=dim) if self.is_MPNET else AENet3()
        self.net = self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=0)
        self.writer = SummaryWriter(self.directory)

        with open(self.directory + 'architecture_config.txt', 'w') as f:
            s = "model:\n" 
            for name, layer in self.net.named_modules():
                s += f'{name}: {str(layer)}\n'
            f.write(s)

    def train(self):
        self.test(0, mode='test')
        self.test(0, mode='val')
        with tqdm(total=self.num_epochs) as pbar:
            for epoch in range(self.num_epochs):
                epoch_running_loss = dict.fromkeys(loss_keys, 0.0)
                with tqdm(total=len(self.loader_train), desc='batch', leave=False) as pbar2:
                    for i, data in enumerate(self.loader_train, 0):
                        x, y = data
                        x = x.to(self.device)
                        y = {k: v.to(self.device) for k, v in y.items()} if self.is_MPNET else x
                        self.optimizer.zero_grad()

                        ### forward + backward + optimize
                        outputs = self.net(x)
                        if not self.is_MPNET:
                            encoding, outputs = outputs[0], outputs[1]
                        loss = self.criterion(outputs, y)

                        loss['loss'].backward()
                        self.optimizer.step()

                        # print statistics
                        for _, (k, val) in enumerate(epoch_running_loss.items()):
                            epoch_running_loss[k] += loss[k].item()
                        pbar2.update(1)
                # print(f'ep={epoch + 1} loss: {epoch_running_loss / i:.3f}')
                for _, (k, val) in enumerate(epoch_running_loss.items()):
                    self.writer.add_scalar('Loss/train_epoch_' + k, val / (i + 1), epoch)

                if (epoch + 1) % 10 == 0:
                    self.test(epoch, mode='test')
                    self.test(epoch, mode='val')
                    state = {
                        'epoch': epoch,
                        'state_dict': self.net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss': epoch_running_loss['loss'] / (i + 1)
                    }
                    torch.save(state, self.directory + f'model_{epoch}.ckpt')
                pbar.set_description('Loss {:.3f}'.format(epoch_running_loss['loss'] / (i + 1)))
                pbar.update(1)
    
    def test(self, epoch, mode='test'):
        net = copy.deepcopy(self.net)
        net.eval()
        if mode == 'test':
            data_loader = self.loader_test
        elif mode == 'val':
            data_loader = self.loader_val
        running_loss = dict.fromkeys(loss_keys, 0.0)
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                x, y = data
                x = x.to(self.device)
                y = {k: v.to(self.device) for k, v in y.items()} if self.is_MPNET else x

                outputs = net(x)
                if not self.is_MPNET:
                    encoding, outputs = outputs[0], outputs[1]
                loss = self.criterion(outputs, y)

                # print statistics
                for _, (k, val) in enumerate(running_loss.items()):
                    running_loss[k] += loss[k].item()
            for _, (k, val) in enumerate(running_loss.items()):
                self.writer.add_scalar('Loss/' + mode + '_epoch_' + k, val / (i + 1), global_step=epoch)