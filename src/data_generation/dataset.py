import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from tqdm import tqdm, trange
import matplotlib.pyplot as plt


class MPNetDataset(Dataset):
    """MPNet Data loader
    """
    def __init__(self, mode='train', mp_df=None):
        path = '_numPlansPerEnv_2500_tabletop'
        thresh_plan = 2000
        thresh_env = 290
        data_folder = 'data/data2500_tabletop/'
        mp_data = pd.read_json(f'{data_folder}MPData{path}.json', orient='index') if mp_df is None else mp_df
        # self.shape_data = pd.read_json(f'{data_folder}MPObsShapeData{path}.json', orient='index')
        self.obs_data = np.load(f'{data_folder}obstacles_encoded.npy').astype(np.float32)
        self.obs_data = self.obs_data.squeeze(-1)
        print('obsdata', self.obs_data.shape)

        dim = 7

        if mode == 'train':
            mp_data = mp_data[(mp_data['plan_id'] < thresh_plan) & (mp_data['env_id'] < thresh_env)]
        elif mode == 'test':
            mp_data = mp_data[(mp_data['plan_id'] >= thresh_plan) & (mp_data['env_id'] < thresh_env)]
        elif mode == 'val':
            mp_data = mp_data[mp_data['env_id'] >= thresh_env]
            
        self.data_state = np.array(mp_data['state'].tolist(), dtype=np.float32)[:,(-2*dim):]
        self.data_action = np.array(mp_data['action'].tolist(), dtype=np.float32)
        print('action shape', self.data_action.shape)
        self.data_envidx = np.array(mp_data['env_id'].tolist())
        print('envidx max', np.max(self.data_envidx))
        print('envidx shape', self.data_envidx.shape)
        self.data_planidx = np.array(mp_data['plan_id'].tolist())
        print('data state', self.data_state.shape)

        self.data_next_pos = self.data_state[:,-dim:] + self.data_action
        assert(len(self.data_state) == len(self.data_planidx) == len(self.data_envidx))

    def __len__(self):
        return self.data_state.shape[0]

    def __getitem__(self, index):
        output = {}
        # y = self.data_next_pos[index, ...]                   # position instead of action
        y = self.data_action[index, ...]
        
        # print('envidx', self.data_envidx[index])
        self.data_envidx[index]
        obs = self.obs_data[self.data_envidx[index], ...]    # pt-cloud encoding
        
        x = np.concatenate((obs, self.data_state[index]))
         
        x = torch.from_numpy(x).type(torch.float32)
        output['action'] = torch.from_numpy(y).type(torch.float32)
        return x, output

class SelfSupervisedDataset(MPNetDataset):
    """MPNet Data loader
    """
    def __init__(self, mode='train', mp_df=None):
        super().__init__()

    def __getitem__(self, index):
        output = {}
        # # y = self.data_next_pos[index, ...]                   # position instead of action
        # y = self.data_action[index, ...]
        
        # obs = self.obs_data[self.data_envidx[index], ...]    # pt-cloud encoding
        
        # x = np.concatenate((obs, self.data_state[index]))
         
        # x = torch.from_numpy(x).type(torch.float32)
        # output['action'] = torch.from_numpy(y).type(torch.float32)
        return x, output


class AEDataset(Dataset):
    """Auto-Encoder data loader
    """
    def __init__(self, mode='train'):
        self.point_cloud = np.load('data/obsdata_tabletop_idx0.npy').astype(np.float32)
        print(self.point_cloud.shape)
        
        num_envs = self.point_cloud.shape[0]
        if mode == 'train':
            self.point_cloud = self.point_cloud[:int(.7 * num_envs),...]
        elif mode == 'test':
            self.point_cloud = self.point_cloud[int(0.7 * num_envs) : int(0.85 * num_envs),...]
        elif mode == 'val':
            self.point_cloud = self.point_cloud[int(.85 * num_envs):,...]
    
    def __len__(self):
        return self.point_cloud.shape[0]

    def __getitem__(self, index):
        x = self.point_cloud[index,...]
        x = np.moveaxis(x, -1, 0)  # get [dim, num_points]
        x = torch.from_numpy(x).type(torch.float32)
        return x, {'foo': 1}  # added `foo` for consistency


def get_train_test_val(is_MPNet=True):
    batch_size = 256 if is_MPNet else 32

    data_folder = 'data/data2500_tabletop/'
    path = '_numPlansPerEnv_2500_tabletop'
    
    # mp_df = pd.read_json(f'data/MPData{path}.json', orient='index')
    mp_df = pd.read_pickle(f'{data_folder}MPData{path}.pkl')
    mp_df = mp_df[mp_df['env_id'] < 355] #TODO: fix

    dataset = MPNetDataset if is_MPNet else AEDataset
    seq = ['train', 'test', 'val']
    loader_seq = []
    for it in seq:
        loader = torch.utils.data.DataLoader(dataset(it, mp_df), batch_size=batch_size,
                                                    shuffle=True, num_workers=10)
        loader_seq.append(loader)
    return loader_seq[0], loader_seq[1], loader_seq[2]