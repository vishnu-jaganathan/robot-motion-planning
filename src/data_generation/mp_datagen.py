from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment, getPtClouds
from architechtures.agent_model import MPNet1
from architechtures.AE_model import AENet3
from data_generation.util import getObstaleEncoding

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

from torch.utils.tensorboard import SummaryWriter


###################################################################
###### Generate data for MPNET
###################################################################
def generateMPData(encoder_model = None):
    '''Generates motion planning data using ompl solver
    '''
    # encoder_model = AENet3()
    # path = 'data/AE_trained_l1_large_64/model_AE.ckpt'
    # encoder_model.load_state_dict(torch.load(path)['state_dict'])
    # encoder_model.eval()

    # hparams = {'get_obstacle_encoding': False, 'num_environments': 550, 'num_motion_plans_per_env': 5000, 'num_obs_samples_per_env':1400}
    num_environments = 550
    num_motion_plans_per_env = 2500
    num_obs_samples_per_env = 2800

    
    get_obstacle_encoding = False
    
    idx = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
    print(idx)

    mpdata = []
    data_obs_encoding = []
    data_obs_info = []
    for env_id in tqdm(range(num_environments), desc='environment'):
        ### sample environment
        stsp = KukaEnv(GUI=False)
        env = GymEnvironment(stsp)
        env.reset(new_obstacle=True)
        env.obs_encoding = None  # to prevent appending this to state
        env.setOMPLSovler()
        
        obsinfo = np.array(env.obstacles)
        info = {'env_id': env_id, 'x':obsinfo[:,1,0].tolist(), 'y':obsinfo[:,1,1].tolist(),
                'z':obsinfo[:,1,2].tolist(), 'dx': obsinfo[:,0,0].tolist(),
                'dy': obsinfo[:,0,1].tolist(),'dz': obsinfo[:,0,2].tolist()}
        data_obs_info.append(info)
        
        ### get obstacle encoding
        if get_obstacle_encoding:
            data_obs_encoding.append(getObstaleEncoding(getPtClouds(env.stspace), encoder_model))
        
        ### Sample start/goal and get different motion plans for the same environment
        for plan_id in tqdm(range(num_motion_plans_per_env), desc='plan', leave=False):
            trajectory = []
            while not trajectory:  # sample until there is a solution
                env.reset(new_obstacle=False)
                trajectory = env.getOptimalActionSeq()

            for traj in trajectory:
                state, next_state, action, reward, done = traj      # state also contains start & goal
                data = {'env_id': env_id, 'plan_id': plan_id, 'state': state.tolist(), 'action': action.tolist()}
                mpdata.append(data)
        env.stspace.disconnect()

        ### save data
        if ((env_id + 1) % 1 == 0) or (env_id == (num_environments - 1)):
            pd.DataFrame(mpdata).to_json(f"data/MPData_{num_motion_plans_per_env}_idx{idx}.json", orient='index')
            pd.DataFrame(data_obs_info).to_json(f"data/MPObsShapeData_{num_motion_plans_per_env}_idx{idx}.json", orient='index')
            # np.save(f"data/MPObsData_{num_motion_plans_per_env}_idx{idx}.npy", np.array(data_obs_encoding, dtype=np.float32))
