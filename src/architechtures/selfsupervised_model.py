import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfSupervisedModel(nn.Module):
    """
    """
    def __init__(self, encoder_model, decoder_model=None, hparams=None):
        super().__init__()

        ptcloud_embed_dim = 128
        action_embed_dim = 32
        encoding_dim = ptcloud_embed_dim + action_embed_dim


        self.encoder = encoder_model
        self.decoder = decoder_model

        self.fc_action_encoder = (
            nn.Linear(7, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, action_embed_dim)
        )

        self.fc_willcollide = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
        
        self.fc_collision_pos = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 2*3)
        )
        
        self.fc_robot_links = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 7*3)
        ) 
    

    def forward(self, x):
        """_summary_

        Args:
            x (dict): input dict containing pointclouds and action taken

        Returns:
            dict: dictionary containing various predcitions 
        """

        out = {} 

        obs_embedding = self.encoder(x['pt_cloud'])     # obstacle embedding
        kinematic_embedding = self.action(x['action'])  # action embedding

        if self.decoder is not None:
            out['decoded_pts'] = self.decoder(obs_embedding)

        encoding = torch.concatenate((obs_embedding, kinematic_embedding), dim=-1)

        out['obs_embedding'] = obs_embedding
        out['kinematic_embedding'] = kinematic_embedding
        out['encoding'] = encoding

        out['willcollide'] = self.fc_willcollide(encoding)     # obstacle encoding
        out['collision_pos'] = self.fc_collision_pos(encoding)
        out['robot_links'] = self.fc_robot_links(encoding)

        return out


