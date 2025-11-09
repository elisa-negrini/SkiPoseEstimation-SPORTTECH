import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import re
import cv2
from random import shuffle
import sys
import json
sys.path.append("..")
from utils import *
from torch.utils.data import DataLoader, Dataset
from utils import *
from sklearn.decomposition import PCA
from einops import rearrange
import random

import pytorch_lightning as pl


class Body25DataModule(pl.LightningDataModule):
    
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    def test_dataloader(self):
        working_dir = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

        dataset_path = os.path.join(working_dir, f'jump{self.FLAGS.testing_jump}', 'body_25_test.json')

        dataset = Body25Dataset(self.FLAGS, dataset_path)

        print("Test the network to reconstruct skis of a ski jumper athlete: \n\n")

        test_loader = DataLoader(dataset, shuffle=False, pin_memory=True,
                                           num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)
        
        return test_loader
        
    
class Body25Dataset(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, FLAGS, data_path):

        self.data_path = data_path 
        self.FLAGS = FLAGS

        # annotations è un file json che contiene un dizionario di liste, l'ho gia pulito nel preprocessing, togliendo 
        # i None e togliendo le multipose, avrò solo una posa per ogni immagine, però devo conservare ogni volts l'informazione
        # sull'immagine in modo da plottarla e vedere risultato

        with open(self.data_path) as f:
            loaddata = json.load(f)
        
        # delete incomplete poses
        if self.FLAGS.filter_openpose == "delete_pose":

            # new dict with valid poses
            new_data = {}

            for key, pose in loaddata.items():
                complete_pose = True
                if pose[0] == [0,0]:
                    pose[0] = pose [1]
                # search for [0,0] joints in the pose
                for joint in pose[:14]:
                    if joint == [0, 0]:
                        complete_pose = False

                # add to the dict only complete poses
                if complete_pose:
                    new_data[key] = pose

            loaddata = new_data
        
        # Dictionary, images as key. Skeleton, offfset and scale as value

        poses_list = [v for v in loaddata.values()]

        poses_list, offsets = translation(poses_list, type='body_25')
        poses_list = rearrange(poses_list, 'b j d -> b (d j)')
        poses_list = torch.tensor(poses_list, dtype=torch.float)
        poses_list, scale = normalize_head(poses_list, type='body_25')
        poses_list = rearrange(poses_list, 'b (d j) -> b j d', d=2)


        for key, pose, offset in zip(loaddata.keys(), poses_list, offsets):
            loaddata[key] = [pose, offset, scale]

        self.data = dict()
        self.data['poses_3d'] = loaddata

        if FLAGS.mode == 'demo':
            if FLAGS.train_dataset == 'ski':
                joints_keep = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
                joints_keep_mapped = [0,1,2,3,4,5,6,7,22,8,9,10,11,12,13]
                joints_masked = [14,15,16,17,18,19,20,21]  # sci destro e sinistro
            
            poses_list_adapted = np.empty(shape=(poses_list.shape[0],FLAGS.n_joints,2))

            poses_list_adapted[:,joints_keep_mapped,:] = poses_list[:,joints_keep,:]

            poses_list_adapted[:,joints_masked,:] = np.zeros(shape=(poses_list.shape[0],1,2))

            poses_list_adapted = torch.tensor(poses_list_adapted, dtype=torch.float)


            for key, pose, offset in zip(loaddata.keys(), poses_list_adapted, offsets):
                loaddata[key] = [pose, offset, scale]

            self.data['poses_3d_new'] = loaddata

    def __len__(self):
        return len(self.data['poses_3d'])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.FLAGS.mode == 'train':
            return list(self.data['poses_3d'].items())[idx]
        else:
            return list(self.data['poses_3d_new'].items())[idx]
        
        
