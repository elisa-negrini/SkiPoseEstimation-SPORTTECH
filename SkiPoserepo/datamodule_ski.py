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

class SkiDataModule(pl.LightningDataModule):
    
    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
    
    def train_dataloader(self):
        # choose the ski dataset on which perform the training
        if not self.FLAGS.fine_tuning:
            if self.FLAGS.train_set_mode == "S2D":
                working_dir = "/home/federico.diprima/Ski_2DPose_Dataset/annotations"
                dataset_path = os.path.join(working_dir, 'train.json')
                print("\nTraining only Ski 2D dataset: ", end="")
            elif self.FLAGS.train_set_mode == "SJ":
                working_dir = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
                dataset_path = os.path.join(working_dir, f'jump{self.FLAGS.testing_jump}' ,'train.json')
                print(f"\nTraining only on Ski Jump dataset (excluded jump{self.FLAGS.testing_jump}): ", end="")
            elif self.FLAGS.train_set_mode == "S2D_SJ":
                working_dir = "/home/federico.diprima/S2D_SJ"
                
                S2D_path = "/home/federico.diprima/Ski_2DPose_Dataset/annotations"
                SJ_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
                with open(os.path.join(S2D_path, "train.json")) as f:
                    S2D = json.load(f)
                with open(os.path.join(SJ_path, f'jump{self.FLAGS.testing_jump}' ,'train.json')) as f:
                    SJ = json.load(f)
                dataset_path = os.path.join(working_dir, 'train.json')
                with open(dataset_path, 'w') as f:
                    json.dump(SJ+S2D, f)
                print("\nTraining S2D_SJ dataset (Ski Jump + Ski 2D): ", end="")
            else:
                # S2D_SJ+S dataset as default
                working_dir = "/home/federico.diprima/S2D_SJ_S"
                dataset_path = os.path.join(working_dir, 'train.json')
                S2D_path = "/home/federico.diprima/Ski_2DPose_Dataset/annotations"
                SJ_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
                SY_path = "/home/federico.diprima/Synthetic_Ski_Jump"
                with open(os.path.join(S2D_path, "train.json")) as f:
                    S2D = json.load(f)
                print(len(S2D))
                with open(os.path.join(SJ_path, f'jump{self.FLAGS.testing_jump}' ,'train.json')) as f:
                    SJ = json.load(f)
                # with open(os.path.join(SJ_path,'train.json')) as f:
                #     SJ = json.load(f)
                print(len(SJ))
                with open(os.path.join(SY_path, "synthetic_processed.json")) as f:
                    SY = json.load(f)
                print(len(SY))
                dataset_path = os.path.join(working_dir, 'train.json')
                with open(dataset_path, 'w') as f:
                    json.dump(S2D+SJ+SY, f)
                print("\nTraining on Final dataset (Ski Jump + Ski 2D + Synthetic data): ", end="")

        else:
            # finetuning only on ski_jump dataset (our usecase)
            working_dir = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
            dataset_path = os.path.join(working_dir,f'jump{self.FLAGS.testing_jump}', 'train.json')
            print("\nLoad checkpoint from", self.FLAGS.load_pretrained_model)
            print("Fine-tuning the pretrained model on Ski Jump dataset: ", end="")


        dataset = SkiDataset(self.FLAGS, dataset_path)
        print(len(dataset), "elements\n\n")


        train_loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                                           num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)
        
        return train_loader
        
    
class SkiDataset(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, FLAGS, data_path):

        self.data_path = data_path
        self.FLAGS = FLAGS
        with open(self.data_path) as f:
            loaddata = json.load(f)
        
        poses_list = [pose for _, pose in loaddata]
        source = [s for s,_ in loaddata]
        
        
        self.data = dict()

        poses_list, _ = translation(poses_list, type = 'ski')
        poses_list = rearrange(poses_list, 'b j d -> b (d j)')
        poses_list = torch.tensor(poses_list, dtype=torch.float)
        poses_list, _ = normalize_head(poses_list, type = 'ski')
        #self.data['ski'] = F.normalize(self.data['ski'], dim=0)
        poses_list = rearrange(poses_list, 'b (d j) -> b j d', d=2)

        self.data['ski'] = []
        for pose, s in zip(poses_list, source):
            self.data['ski'].append([pose, s])

    def __len__(self):
        return len(self.data['ski'])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict ()
        
        if self.FLAGS.mode == "train" and self.FLAGS.data_augmentation:

            # Data Augmentation

            # AXIAL SYMMETRY Y (X=0)
            if(np.random.randint(low=0, high=2, size=(1,))):
                pose = self.data['ski'][idx][0]
                joints_symmetric = torch.clone(pose)
                joints_symmetric[:, 0] = -joints_symmetric[:, 0]
                sample['msk'] = [joints_symmetric, self.data['ski'][idx][1]]
            # ROTATION
            if(np.random.randint(low=0, high=2, size=(1,))):
                pose = self.data['ski'][idx][0]
                angle = random.uniform(0, 20)
                R = torch.tensor(theta_rotation(np.radians(angle)),dtype=torch.float)
                joints_flipped = torch.matmul(pose, R)
                sample['msk'] = [joints_flipped, self.data['ski'][idx][1]]

            else:
                sample['msk'] = self.data['ski'][idx]
        
        else:
            sample['msk'] = self.data['ski'][idx]

        return sample['msk']
