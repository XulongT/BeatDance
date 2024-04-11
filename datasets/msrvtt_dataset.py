import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSRVTTDataset(Dataset):
    """
        videos_dir: directory where all videos are stored 
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """
    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.data_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.frames = config.num_frames
        self.video_features_dir = os.path.join(self.data_dir, 'video_feature')
        self.music_features_dir = os.path.join(self.data_dir, 'music_feature')
        self.video_beats_dir = os.path.join(self.data_dir, 'video_beat')
        self.music_beats_dir = os.path.join(self.data_dir, 'music_beat')
        self.beat_dim = 10

        if split_type == 'train':
            self.data = open(os.path.join(self.data_dir, 'train.txt')).readlines()
        elif split_type == 'val':
            self.data = open(os.path.join(self.data_dir, 'val.txt')).readlines()
        elif split_type == 'test':
            self.data = open(os.path.join(self.data_dir, 'test.txt')).readlines()

            
    def __getitem__(self, index):
        video_feature_dir = os.path.join(self.video_features_dir, self.data[index].replace('\n', '.pt'))
        music_feature_dir = os.path.join(self.music_features_dir, self.data[index].replace('\n', '.pt'))
        video_feature = torch.load(video_feature_dir)
        music_feature = torch.load(music_feature_dir)
        video_avg_feature = self.video_preprocess(video_feature)
        music_avg_feature = self.music_preprocess(music_feature)
        data = {
                'video_id': index,
                'video': video_avg_feature,
                'music': music_avg_feature,
                'filename': self.data[index].replace('\n', '.pt'),
            }
        video_beat_dir = os.path.join(self.video_beats_dir, self.data[index].replace('\n', '.pt'))
        music_beat_dir = os.path.join(self.music_beats_dir, self.data[index].replace('\n', '.pt'))
        video_beat = torch.load(video_beat_dir)
        music_beat = torch.load(music_beat_dir)


        video_beat_avg_feature = self.video_beat_preprocess(video_beat)
        music_beat_avg_feature = self.music_beat_preprocess(music_beat)

        data['video_beat'] = video_beat_avg_feature
        data['music_beat'] = music_beat_avg_feature

        return data
    
    def __len__(self):
        return len(self.data)


    def beat_align(self, beat):
        max_len = 100
        beat_index = torch.nonzero(beat).flatten()
        beat_index = torch.nn.functional.pad(beat_index, (0, (max_len - beat_index.shape[0])), value=0)
        return beat_index

    # to make video sequences evenly distributed in 30 portions
    def video_preprocess(self, video_feature):
        fps = video_feature['fps']
        feature = video_feature['video_clip_feature']
        avg_pool = torch.zeros((self.frames, feature.shape[-1]))
        split_feature = torch.chunk(input=feature, chunks=self.frames, dim=0)
        for i in range(len(split_feature)):
            avg_pool[i, :] = torch.mean(split_feature[i], dim=0)
        return avg_pool

    # to make music sequences evenly distributed in 30 portions
    def music_preprocess(self, music_feature):
        fps = music_feature['fps']
        feature = music_feature['music_bert_feature']
        avg_pool = torch.zeros((self.frames, feature.shape[-1]))
        split_feature = torch.chunk(input=feature, chunks=self.frames, dim=0)
        for i in range(len(split_feature)):
            avg_pool[i, :] = torch.mean(split_feature[i], dim=0)
        return avg_pool

    # to make video beat sequences evenly distributed in 30 portions
    def video_beat_preprocess(self, video_beat_feature):
        fps = video_beat_feature['fps']
        feature = video_beat_feature['video_beat'][:100]
        return feature.reshape((self.frames, self.beat_dim)).float()

    # to make music beat sequences evenly distributed in 30 portions
    def music_beat_preprocess(self, music_beat_feature):
        fps = music_beat_feature['fps']
        feature = music_beat_feature['music_beat'][:100]
        return feature.reshape((self.frames, self.beat_dim)).float()
