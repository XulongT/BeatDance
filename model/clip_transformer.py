import torch
import torch.nn as nn
from config.base_config import Config
from modules.transformer import PoseTransformer
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, beat_similarity, qb_norm
from torch import nn
import torch

import torch

# class PositionalEncoding(torch.nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()

#         # Create constant positional encoding matrix
#         self.pe = torch.nn.Parameter(torch.zeros(max_len, d_model))
#         # torch.nn.init.xavier_uniform_(self.pe)  # Initialize pe with Xavier uniform initialization

#     def forward(self, x):
#         # print('x', x)
#         # print('pe', self.pe[:x.size(1), :])
#         x = x + self.pe[:x.size(1), :]
#         # print('x', x)
#         return x


class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config

        config.pooling_type = 'avg'
        self.dropout1 = config.dropout1
        self.dropout2 = config.dropout2

        video_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        music_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        self.music_transformer = nn.TransformerEncoder(video_encoder_layer, num_layers=config.num_layers)
        self.video_transformer = nn.TransformerEncoder(music_encoder_layer, num_layers=config.num_layers)

        self.music_linear = nn.Linear(768, config.embed_dim)
        self.video_linear = nn.Linear(512, config.embed_dim)

        self.clip_logit_scale = torch.FloatTensor([4.6052]).cuda()

        video_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        music_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        self.music_beat_transformer = nn.TransformerEncoder(video_beat_encoder_layer, num_layers=config.num_layers)
        self.video_beat_transformer = nn.TransformerEncoder(music_beat_encoder_layer, num_layers=config.num_layers)
        self.music_beat_linear = nn.Sequential(
            nn.Linear(10, 64),
            nn.Linear(64, config.embed_dim),
        )
        self.video_beat_linear = nn.Sequential(
            nn.Linear(10, 64),
            nn.Linear(64, config.embed_dim),
        )


        self.l1, self.l2= nn.Linear(512, 256), nn.Linear(512, 256)

        self.music_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)
        self.video_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)

    def forward(self, data, phase='test'):
        batch_size = data['video'].shape[0]
        music_data = data['music']
        video_data = data['video']

        music_data = (self.music_linear(music_data))
        video_data = (self.video_linear(video_data))


        music_beat = data['music_beat']
        video_beat = data['video_beat']
        music_beat = self.music_beat_linear(music_beat)
        video_beat = self.video_beat_linear(video_beat)


        video_features_trans = self.video_multimodel_fuse(video_data, video_beat)
        music_features_trans = self.music_multimodel_fuse(music_data, music_beat)

        music_features_trans['music_fuse'] = music_features_trans['music_fuse'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        music_features_trans['music_beat'] = music_features_trans['music_beat'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        video_features_trans['video_fuse'] = video_features_trans['video_fuse'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        video_features_trans['video_beat'] = video_features_trans['video_beat'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))     
        
        return music_features_trans, video_features_trans   


    def music_multimodel_fuse(self, music_data, music_beat):

        music_data = self.music_transformer(music_data)
        music_beat = self.music_beat_transformer(music_beat)
        add_data = music_data + music_beat
        mul_data = music_data * music_beat
        fuse_data = self.l1(torch.cat([add_data, mul_data], dim=-1))
        music_beat, _ = self.music_transformer_fuse(music_beat, music_data, music_data)
        fuse_data = {'music_data': music_data, 'music_beat': music_beat, 'music_fuse': fuse_data}
        return fuse_data
        

    def video_multimodel_fuse(self, video_data, video_beat):
        video_data = self.video_transformer(video_data)
        video_beat = self.video_beat_transformer(video_beat)
        add_data = video_data + video_beat
        mul_data = video_data * video_beat
        fuse_data = self.l2(torch.cat([add_data, mul_data], dim=-1))
        video_beat, _ = self.video_transformer_fuse(video_beat, video_data, video_data)
        fuse_data = {'video_data': video_data, 'video_beat': video_beat, 'video_fuse': fuse_data}           
        return fuse_data
