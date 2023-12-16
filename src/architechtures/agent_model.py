import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



###################################################################
###### MPNet architechtures
###################################################################
class MPNet1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MPNet1, self).__init__()

        self.fc = nn.Sequential( # larger model
            nn.Linear(state_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 384), nn.LeakyReLU(),
            nn.Linear(384, 384), nn.LeakyReLU(),
            nn.Linear(384, 256), nn.LeakyReLU(), 
            nn.Linear(256, 256), nn.LeakyReLU(), 
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        out = {}
        output = self.fc(state)
        # out['action'] = output # if training 
        out = output
        return out