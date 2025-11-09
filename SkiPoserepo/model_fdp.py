from einops import rearrange
import json
import pytorch_lightning as pl
import wandb
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import datetime
from transformer import Transformer
import sys
sys.path.append("..")
import pickle as pkl
from SkiPose.utils_fdp import *
import cv2
import matplotlib.pyplot as plt
import os
from plot import plot, plot_cv2, plot_body
from SkiPose.utils_fdp import invert_translation, invert_normalize_head, align_ski

class AdaptationNetwork(LightningModule):
    def __init__(
        self,
        FLAGS,
        # config_wandb,
        **kwargs,
    ):
        super().__init__()
        self.FLAGS = FLAGS
        if self.FLAGS.dataset == 'ski':
            self.mapping = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,22], [22,8], [22,11], [8,9], 
                    [9,10], [10,15], [10,16], [14,15], [15,16], [16,17], [11,12], [12,13], [13,19], 
                    [13,20], [18,19], [19,20], [20,21]]
        if self.FLAGS.dataset == 'body_25':
            self.mapping = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], 
                    [11,22], [11,24], [22,23], [8, 12], [12, 13], [13, 14], [14,21], [14,19], [19,20],  
                    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18]]
        # output format
        if self.FLAGS.train_dataset == 'ski':
            self.train_mapping = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,22], [22,8], [22,11], [8,9], 
                    [9,10], [10,15], [10,16], [14,15], [15,16], [16,17], [11,12], [12,13], [13,19], 
                    [13,20], [18,19], [19,20], [20,21]]
        
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            Transformer(128, 6, 16, 64, 128, 0.),
            nn.Linear(128, 2)
        )
        self.can_plot = True
        self.new_joints = []
        self.inputs = []
        self.pos_enc = nn.Parameter(torch.randn(1,2*self.FLAGS.n_joints),requires_grad=True)
        
        annotation_path= "/home/federico.diprima/Ski_Jump_Annotated/annotations"
        with open(os.path.join(annotation_path,f'jump{self.FLAGS.testing_jump}',"test.json")) as f:
            self.gt_annotation = json.load(f)
        
        self.image_shape = (720,1280)
        self.total_mpjpe_sum = 0
        self.total_mpjpe_normalized_sum = 0
        self.skis_mpjpe_sum = 0
        self.skis_mpjpe_normalized_sum = 0
        self.n_samples = 0
        
        # PLOT
        self.gt = []
        self.pred = []
        self.epoch = 0
        self.adapted = []
        self.step = 0

    def forward(self, z, encoder,FLAGS):
        if FLAGS.mode == 'train':
            if FLAGS.masking_mode == 'ski':
                indexes = torch.tensor([14,15,16,17,18,19,20,21])
            if FLAGS.masking_mode == 'random':
                indexes = torch.tensor(np.sort(np.random.choice(range(0,self.FLAGS.n_joints-1), self.FLAGS.masked_joints, replace=False)))

        else:
            #Choose Train and Test Dataset in DEMO mode
            if FLAGS.train_dataset == 'ski':
                if FLAGS.dataset == 'body_25':
                    indexes = torch.tensor([14,15,16,17,18,19,20,21])

        keep_index = []
        z[:,indexes,:] = torch.zeros(z[:,indexes,:].size(), device="cuda") 
        z = rearrange(z, 'b j d -> b (d j)')
        z += self.pos_enc.cuda()
        z = rearrange(z, 'b (d j) -> b j d', d=2)
        for i in range(self.FLAGS.n_joints):
            if i not in indexes:
                keep_index.append(i)
                
        bone_map = []
        bone_map_id = []
        for idx,bone in enumerate(self.mapping):
            #for index in rand_index:
            for index in indexes:
                if index in bone:
                    bone_map.append(bone)
                    bone_map_id.append(idx)
                    break

        return encoder(z), indexes, keep_index, bone_map, bone_map_id

    def training_step(self, batch, batch_idx):
        loss_fn = nn.MSELoss()
        joints = batch[0]
        source = batch[1].tolist()

        z = joints.clone()

        self.generated, self.mask_index, self.no_mask_index, self.bone_map, self.bone_map_id = self(z, self.encoder,self.FLAGS)

        self.total_generated = torch.empty(joints.size()).bfloat16().cuda()

        self.total_generated[:,self.mask_index,:] = self.generated[:,self.mask_index,:]
        self.total_generated[:,self.no_mask_index,:] = joints[:,self.no_mask_index,:].bfloat16()
        

        joints_mask = joints[:,self.mask_index,:]
        joints_mask = rearrange(joints_mask, 'b j d -> b (d j)')

        joints_gen = self.generated[:,self.mask_index,:]
        joints_gen = rearrange(joints_gen, 'b j d -> b (d j)')

        if self.FLAGS.weighted_loss and self.FLAGS.train_set_mode == 'total':

            # dataset lengths
            len_ski_jump = 350
            len_ski_2d = 1982

            # dataset weights
            weight_ski_jump = 1.0
            weight_ski_2d = len_ski_jump / len_ski_2d


            loss = torch.nn.functional.mse_loss(joints_gen, joints_mask, reduction='none')
            weighted_loss = torch.zeros_like(loss)

            #ciclare sulla batch x loss pesato in base allea source (1=ski_jump, 0=ski_2d)
            for i in range(len(joints)):
                if source[i] == 1:
                    weighted_loss[i] = loss[i] * weight_ski_jump
                else:
                    weighted_loss[i] = loss[i] * weight_ski_2d
            
            mse_loss = torch.mean(weighted_loss)
    
        else:
            mse_loss = loss_fn(joints_gen,joints_mask)

        self.log(" Train MSE loss", mse_loss, prog_bar=True)

        self.gt = joints[-1].tolist()
        self.pred = self.total_generated[-1].tolist()

        return mse_loss

    def test_step(self, batch, batch_idx):
        mse_loss = nn.MSELoss()
        
        images = batch[0]
        joints = batch[1][0]
        offsets = batch[1][1]
        scale = batch[1][2][0]

        z = joints.clone()

        self.generated, self.mask_index, self.no_mask_index, self.bone_map, self.bone_map_id = self(z, self.encoder,self.FLAGS)

        # shape self.generated 'b j d', batch x joints x d  d=2

        self.total_generated = torch.empty(joints.size()).bfloat16().cuda()

        self.total_generated[:,self.mask_index,:] = self.generated[:,self.mask_index,:]
        self.total_generated[:,self.no_mask_index,:] = joints[:,self.no_mask_index,:].bfloat16()

        
        # Invert normalization and translation to plot the results

        self.total_generated = rearrange(self.total_generated, 'b j d -> b (d j)')
        self.total_generated = invert_normalize_head(self.total_generated, scale)
        self.total_generated = rearrange(self.total_generated, 'b (d j) -> b j d', d=2)

        self.total_generated = self.total_generated.tolist()
        offsets = offsets.tolist()
        self.total_generated = invert_translation(self.total_generated, offsets)
        
        # function to have skiies aligned
        if self.FLAGS.aligned_skis:
            for pose in self.total_generated:
                pose = align_ski(pose)
        
        # Ground truth poses
        gt = np.empty((self.FLAGS.batch_size, self.FLAGS.n_joints, 2))
        for i,image in enumerate(images):
            height = self.image_shape[0]
            width = self.image_shape[1]
            if image in self.gt_annotation.keys():
                pose = np.array(self.gt_annotation[image])
                pose = [[x * width, y * height] for x, y in pose]
                gt[i,:,:] = pose
                self.n_samples+=1
        

        print("Batch:", self.step+1)

        heights = np.max(gt[:, :, 1], axis=1) - np.min(gt[:, :, 1], axis=1)
        ski_length = np.linalg.norm(gt[:, 21] - gt[:, 18], axis=1)


        # MPJPE total figure
        euclidean_distances = np.linalg.norm(self.total_generated - gt, axis=2)
        mpjpe_per_sample = np.mean(euclidean_distances, axis=1)
        mpjpe_batch = np.mean(mpjpe_per_sample)
        self.total_mpjpe_sum += sum(mpjpe_per_sample)
 
        print("Batch MPJPE:", round(mpjpe_batch,3))
    
        normalized_mpjpe_per_sample = np.mean(euclidean_distances, axis=1) / heights
        mpjpe_normalized_batch = np.mean(normalized_mpjpe_per_sample)
        self.total_mpjpe_normalized_sum += sum(normalized_mpjpe_per_sample)

        print("Batch Normalized MPJPE : ", round(mpjpe_normalized_batch,3))


        #MPJPE skis (only skies joints)
        skis_gt = gt[:, 14:22, :]
        skis_predicted = self.total_generated[:, 14:22, :]

        skis_euclidean_distances = np.linalg.norm(skis_predicted - skis_gt, axis=2)
        skis_mpjpe_per_sample = np.mean(skis_euclidean_distances, axis=1)
        skis_mpjpe_batch = np.mean(skis_mpjpe_per_sample)
        self.skis_mpjpe_sum += sum(skis_mpjpe_per_sample)

        print("Skis batch MPJPE:", round(skis_mpjpe_batch,3))

        normalized_skis_mpjpe_per_sample = np.mean(skis_euclidean_distances, axis=1) / ski_length        
        skis_mpjpe_normalized_batch = np.mean(normalized_skis_mpjpe_per_sample)
        self.skis_mpjpe_normalized_sum += sum(normalized_skis_mpjpe_per_sample)

        print("Skis batch Normalized MPJPE:", round(skis_mpjpe_normalized_batch,3))


        # Plot one sample for each step
        final_pose = self.total_generated[-1].tolist()
        gt = gt[-1].tolist()
        image = images[-1]
        self.step += 1

        plot_cv2(final_pose, image, self.train_mapping, self.FLAGS.test_result_dir, f"example{self.step}.jpg")
        plot_body(final_pose, image, self.train_mapping, self.FLAGS.test_result_dir, f"example{self.step}_body.jpg")
        plot_cv2(gt, image, self.train_mapping, self.FLAGS.test_result_dir, f"example{self.step}_GT.jpg")
        
        

    def configure_optimizers(self):
        opt_encoder = torch.optim.Adam(self.parameters(), lr=self.FLAGS.lr)
        scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_encoder, gamma=0.95)

        return [opt_encoder], [scheduler_encoder]
    
    def on_train_epoch_end(self):
        self.epoch += 1
        plot(self.pred, self.gt, self.mapping, self.FLAGS.train_result_dir, self.epoch)            

    def on_test_epoch_end(self):
        total_mpjpe = self.total_mpjpe_sum/self.n_samples
        print("Total MPJPE:", round(total_mpjpe,3))

        total_nomalized_mpjpe = self.total_mpjpe_normalized_sum/self.n_samples
        print("Total Normalized MPJPE:", round(total_nomalized_mpjpe,3))

        skis_mpjpe = self.skis_mpjpe_sum/self.n_samples
        print("Skis MPJPE", round(skis_mpjpe,3))

        skis_normalized_mpjpe = self.skis_mpjpe_normalized_sum/self.n_samples
        print("Skis Normalized MPJPE", round(skis_normalized_mpjpe,3))


        with open(os.path.join(self.FLAGS.test_result_dir,'MPJPE.txt'), 'w') as file:
            file.write(f"Total MPJPE: {round(total_mpjpe,3)} \nSkis MPJPE: {round(skis_mpjpe,3)}\n")
            file.write(f"Total Normalized MPJPE: {round(total_nomalized_mpjpe,3)} \nSkis Normalized MPJPE: {round(skis_normalized_mpjpe,3)}")
