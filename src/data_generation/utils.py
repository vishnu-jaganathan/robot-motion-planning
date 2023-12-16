import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def getObstaleEncoding(ptclouds, encoder_model=None, return_recons=False):
    """Generates obstacle pt-cloud data and obststacle embedding vector.
    """    
    encoder_inp = torch.from_numpy(np.array(ptclouds)).type(torch.float32)
    encoder_inp = torch.moveaxis(encoder_inp, 0, 1).unsqueeze(0)
    with torch.no_grad():
        encoding, d_out = encoder_model(encoder_inp)
        encoding = encoding.cpu().data.numpy().flatten()
    
    if return_recons:
        d_out = d_out.squeeze(0).cpu().data.numpy()
        d_out = np.moveaxis(d_out, 0, 1)
        return encoding, d_out, encoder_inp
    return encoding

def combineDataframes():
    """Combines multiple json or npy data files 
    """
    base_name = '_tabletop_AE_idx'
    data_dir = 'data/' + 'selfsupervided_200perEnv/processed/'
    typenames = [f'MPRobotData{base_name}', f'MPObsShapeData{base_name}']
    
    all_file_names = os.listdir(data_dir)
    print(all_file_names)
    for typename in typenames:
        data_files = []
        first_envid = 0
        file_names = [it for it in all_file_names if typename in it]
        file_names.sort()
        print(file_names)
        if '.json' in file_names[0]:
            formattype = 'json'
        elif '.npy' in file_names[0]:
            formattype = 'npy'  
        print(file_names)
        
        for i in tqdm(range(len(file_names))):
            # print('reading:\t', file_names[i], end=' ')
            if formattype == 'json':
                data_file = pd.read_json(data_dir + file_names[i], orient='index')
                data_file['env_id'] += first_envid
                first_envid = data_file.iloc[-1]['env_id'] + 1
            elif formattype == 'npy':
                data_file = np.load(data_dir + file_names[i], allow_pickle=False)
            data_files.append(data_file)
        
        path = f'{data_dir}{typename}combined.{formattype}'
        if os.path.exists(path):
            ValueError(f"Error!! File already exists: {path}")
        
        if formattype == 'json':
            combined_df = pd.concat(data_files, ignore_index=True)
            print(combined_df, '\nWAIT!!!')
            combined_df.to_json(path, orient='index')
        elif formattype == 'npy':
            combined_data = np.concatenate(data_files, axis=0, dtype=np.float32)
            print(combined_data.shape, '\nWAIT!!!')
            np.save(path, combined_data)
    print('Done!')