from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment
from architechtures.agent_model import MPNet1
from data_generation.dataset import  MPNetDataset

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import datetime

from tqdm import tqdm, trange
import os
import copy
import time
import matplotlib.pyplot as plt


def visualizeMPNetOnDataset(mode='val'):
    """Visualize/Evaluate the MPNET performance on the dataset
    """
    hparams = {'use_MPNet': True}
   
    data_folder = 'data/data2500_tabletop/'
    path = '_numPlansPerEnv_2500_tabletop'
    mp_df = pd.read_pickle(f'{data_folder}MPData{path}.pkl')
    mp_df = mp_df[mp_df['env_id'] < 355]
    dataset = MPNetDataset(mode, mp_df)
    del mp_df

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    dim = 7
    obs_dim = 128
    mpnet_model = MPNet1(state_dim=obs_dim + 2*dim, action_dim=dim).to(device)

    path2 = 'logs/MPNet_baseline_128PointNet/model_69.ckpt'
    mpnet_model.load_state_dict(torch.load(path2)['state_dict'])
    mpnet_model.eval()

    
    ### load dataset
    path = f'{data_folder}MPObsShapeData{path}.json'
    shape_data = pd.read_json(path, orient='index')
    data_env_id = dataset.data_envidx
    data_plan_id = dataset.data_planidx

    stats = {'num_trivial': 0, 'num_solved': 0, 'num_total': 0}
    pre_envid = -1
    pre_planid = -1
    pbar = trange(0, dataset.__len__())
    for i in pbar:
        envid = data_env_id[i]
        if envid != pre_envid:
            if pre_envid != -1:
                env.stspace.disconnect()
            pre_envid = envid

            stsp = KukaEnv(GUI=False)
            env = GymEnvironment(stsp)
            obs_info = shape_data.iloc[envid]
            env.populateObstacles(obs_info)
        
        if data_plan_id[i] != pre_planid:
            pre_planid = data_plan_id[i]
            x, y = dataset.__getitem__(i)
            env.stspace.init_state = x[-dim:].cpu().data.numpy().flatten()
            env.stspace.goal_state = x[(-2*dim):-dim].cpu().data.numpy().flatten()

            stats['num_total'] += 1
            state = env.reset(sampleStandGoal=False, new_obstacle=False)
            if env.stspace._edge_fp(env.state, env.stspace.goal_state):
                stats['num_trivial'] += 1
            
            ### plan using trained model
            for t in range(20):
                # input('press enter')
                if hparams['use_MPNet']:
                    with torch.no_grad():
                        encoding = x[:obs_dim]
                        mpnet_inp = np.concatenate((encoding, state))
                        # mpnet_inp = state
                        mpnet_inp = torch.from_numpy(mpnet_inp).type(torch.float32).to(device)
                        action = mpnet_model(mpnet_inp)
                        action = action.cpu().data.numpy().flatten()
                else:
                    _, y = dataset.__getitem__(i + t)
                    action = y['next_pos'].cpu().data.numpy().flatten()
                state, r, done, _ = env.step(action)
                
                if r == -1: # if collision detected
                    break
                    # pass  # in case stochastic policy
                if done:
                    stats['num_solved'] += 1
                    break
        if i%100 == 0:
            pbar.set_description('solved/total: {}/{} ({} trivial)'.format(stats['num_solved'],
             stats['num_total'], stats['num_trivial']))