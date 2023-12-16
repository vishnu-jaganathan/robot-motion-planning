from environment.kuka_env import KukaEnv
from environment.envwrapper import GymEnvironment, getPtClouds

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import datetime
import sys

from tqdm import tqdm, trange
import os
import copy
import time
import matplotlib.pyplot as plt


def genPtcloudTabletop():
    """Generates point cloud from depth camera
    """
    stspace = KukaEnv(GUI=False)
    env = GymEnvironment(stspace)
    num_envs = 30000

    idx = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
    print(idx)

    pt_clouds = []
    data_obs_info = []
    for env_id in tqdm(range(num_envs)):
        env.reset()
        
        obsinfo = np.array(env.obstacles)
        info = {'env_id': env_id, 'x':obsinfo[:,1,0].tolist(), 'y':obsinfo[:,1,1].tolist(),
                'z':obsinfo[:,1,2].tolist(), 'dx': obsinfo[:,0,0].tolist(),
                'dy': obsinfo[:,0,1].tolist(),'dz': obsinfo[:,0,2].tolist()}
        data_obs_info.append(info)
        pt_clouds.append(getPtClouds(env.stspace))

        if ((env_id + 1) % 1000 == 0) or (env_id == (num_envs - 1)):
            pd.DataFrame(data_obs_info).to_json(f"data/MPObsShapeData_tabletop_AE_idx{idx}.json", orient='index')
            np.save(f"data/obsdata_tabletop_idx{idx}.npy", np.array(pt_clouds, dtype=np.float32))

def getPtCloudFromFile():
    """Obtain the pt-cloud or encoding data by creating environment from an `ObsShapeData` file.
    """
    stspace = KukaEnv(GUI=False)
    env = GymEnvironment(stspace)
    env.reset()
    
    shape_data = pd.read_json('data/MPObsShapeData_numPlansPerEnv_2500_tabletop.json', orient='index')
    idx = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
    print(idx)

    num_envs = shape_data['env_id'].iloc[-1]
    pt_clouds = []
    for env_id in tqdm(range(num_envs)):
        env.reset(sampleStandGoal=False, new_obstacle=False)
        obs_info = shape_data.iloc[env_id]
        env.populateObstacles(obs_info, removelastIdx=2) # remove walls for getting pt-clouds

        pt_clouds.append(getPtClouds(env.stspace))
        # encoding, d_out, encoder_inp = getObstaleEncoding(getPtClouds(env.stspace), aencoder_model, return_recons=True)

        if ((env_id + 1) % 1000 == 0) or (env_id == (num_envs - 1)):
            # np.save(f"data/obsdata_tabletop_idx{idx}.npy", np.array(pt_clouds, dtype=np.float32))
            pd.DataFrame(pt_clouds).to_json(f"data/trash{idx}.json", orient='index')


def genSelfSupervisedData():
    """Generate self-supervised data for training the self-supervised model
    """
    
    stspace = KukaEnv(GUI=False)
    env = GymEnvironment(stspace)
    data_path = 'data/current/'
    num_envs = 30000 #3000
    num_actions_per_env = 200 #2000
    max_actions_before_reset = 5 #20  # start in a new part of the environment (if have not collided so far)
    num_feas_acts = num_actions_per_env // 2
    
    assert(max_actions_before_reset <= num_actions_per_env)
    max_action_norm = 0.8 * np.sqrt(env.dim)

    idx = datetime.datetime.now().strftime("%m%d_%H_%M_%S_%f")
    print(idx)

    pt_clouds = []
    data_obs_info = []
    data_robot_info = []
    for env_id in tqdm(range(num_envs)):
        env.reset()

        # take random actions
        for act_id in tqdm(range(num_actions_per_env), desc='action', leave=False):
            desired_collision = True    # to blance the data
            if act_id < num_feas_acts:
                desired_collision = False
            collided = not desired_collision

            while collided != desired_collision: 
                action = np.random.uniform(low=env.stspace.bound[:env.dim], high=env.stspace.bound[env.dim:], size=(env.dim,))            
                if np.linalg.norm(action) > max_action_norm:
                    action *= max_action_norm / np.linalg.norm(action)
                current_state = env.state
                new_state, reward, done, _  = env.step(action)
                collided = (reward == -1) #TODO cast to int

            collision_pos = np.zeros((2,3), dtype=np.float32)
            if collided:
                collision_pos = np.array(env.stspace.get_collision_position(), dtype=np.float32)
            robot_links = np.array(env.stspace.get_links_position(), dtype=np.float32)

            ### get action info
            info = {'env_id': env_id, 'act_id':act_id, 'curent_state': current_state.tolist(), 'action': action.tolist(),
                    'collided':collided, 'collision_pos': collision_pos.tolist(), 'robot_links': robot_links.tolist()}
            data_robot_info.append(info)
            
            ## get obstacle info
            if act_id == 0:
                obsinfo = np.array(env.obstacles)
                info = {'env_id': env_id, 'x':obsinfo[:,1,0].tolist(), 'y':obsinfo[:,1,1].tolist(),
                        'z':obsinfo[:,1,2].tolist(), 'dx': obsinfo[:,0,0].tolist(),
                        'dy': obsinfo[:,0,1].tolist(),'dz': obsinfo[:,0,2].tolist()}
                data_obs_info.append(info)
                pt_clouds.append(getPtClouds(env.stspace))

            if collided or ((act_id + 1) % (num_actions_per_env // max_actions_before_reset)) == 0:
                env.reset(new_obstacle=False)
        
        if ((env_id + 1) % (1 + num_envs//500) == 0) or (env_id == (num_envs - 1)):
            pd.DataFrame(data_obs_info).to_json(f"{data_path}MPObsShapeData_tabletop_AE_idx{idx}.json", orient='index')
            np.save(f"{data_path}obsdata_tabletop_idx{idx}.npy", np.array(pt_clouds, dtype=np.float32))
            pd.DataFrame(data_robot_info).to_json(f"{data_path}MPRobotData_tabletop_AE_idx{idx}.json", orient='index')
    
    print(pd.DataFrame(data_obs_info))
    print(pd.DataFrame(data_robot_info))

    