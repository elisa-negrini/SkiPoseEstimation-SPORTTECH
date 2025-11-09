from einops import rearrange
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
import wandb
import numpy as np
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import datetime
from transformer import Transformer, PatchEmbed
from timm.models.layers import trunc_normal_
import sys
sys.path.append("..")
import pickle as pkl
from utils import *
from torchvision.io import read_image
from torchvision import models
import random


class SkiDat_Network(LightningModule):
    def __init__(
        self,
        FLAGS,
        # config_wandb,
        **kwargs,
    ):
        super().__init__()
        self.FLAGS = FLAGS

        self.body25_mapping = range(self.FLAGS.n_joints)
        self.body19_mapping = [0,1,2,3,4,5,7,8,9,11,12,13,16,17,18,19,20,23,24]

        self.remove_joints = [6,10,14,15,21,22]


        tiny = [192, 12, 3, 64, 768, 0.]
        small = [384, 12, 6, 64, 1536, 0.]
        base = [768, 12, 12, 64, 3072, 0.]
        large = [1024, 24, 16, 64, 4096, 0.]

        dat = [128, 6, 16, 64, 128, 0.]

        
        # dim, depth, heads, dim_head, mlp_dim, dropout
        tiny_4Xheads = [192, 12, 12, 16, 768, 0.]

        self.C = dat

        if self.FLAGS.use_image_patches:

            self.patch_embed = PatchEmbed(img_size=self.FLAGS.img_size, patch_size=self.FLAGS.patch_size, 
                                        in_chans=self.FLAGS.in_chans, embed_dim=self.C[0])
            num_patches = self.patch_embed.num_patches

        if self.FLAGS.use_backbone:

            self.init_backbone()


        self.encoder = nn.Sequential(
            nn.Linear(2, self.C[0]),
            nn.LeakyReLU()
        )

        self.transformer = Transformer(self.C[0], self.C[1], self.C[2], self.C[3], self.C[4], self.C[5])

        self.decoder = nn.Linear(self.C[0], 2)

        self.img_num = 0
        self.img_set = set()

        self.can_plot = True
        
        self.new_joints = []
        self.inputs = []

        # self.pos_drop = nn.Dropout(p=self.FLAGS.drop_rate)

        if self.FLAGS.use_backbone:
            self.pos_enc = nn.Parameter(torch.randn(1,self.FLAGS.n_joints+1,self.C[0]),requires_grad=True)
        else:
            self.pos_enc = nn.Parameter(torch.randn(1,self.FLAGS.n_joints,self.C[0]),requires_grad=True)

        if self.FLAGS.use_image_patches:
            self.pos_enc_image = nn.Parameter(torch.zeros(1, num_patches , self.C[0]))

        # self.mse_loss = nn.MSELoss()
        self.mse_loss = nn.L1Loss()

    def init_backbone(self):
        # Pretrained SwAV backbone
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
        self.feature_extractor = SwAV.load_from_checkpoint(weight_path, strict=True)
        layers = list(self.feature_extractor.model.children())[:-2]
        self.backbone_output_shape = list(layers[-2][-1].children())[-2].num_features
        self.feature_extractor = nn.Sequential(*layers,nn.Conv2d(2048, self.C[0], kernel_size=(1, 1), stride=(1, 1), bias=False))
        del layers, weight_path

        if self.FLAGS.freeze_fe:
            for par in self.feature_extractor.parameters():
                par.requires_grad=False
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()

    def _forward_features(self, x):
        #x = self.feature_extractor(x)
        representations = self.feature_extractor(x).flatten(1)
        return representations

    def forward(self, z, x = None, mapping = None,):

        if self.FLAGS.mode == 'train':
            if mapping == None:
                    # total_rand_index = torch.tensor(np.sort(np.random.choice(range(1,self.FLAGS.n_joints), self.FLAGS.masked_joints, replace=False)))
                    total_rand_index = torch.tensor([6,10,14,15,16,17,21,22,23,24])
                    z[:,total_rand_index,:] = torch.zeros(z[:,total_rand_index,:].size(), device="cuda")
            else:
                # rand_index = torch.tensor(np.sort(np.random.choice(mapping, self.FLAGS.masked_joints, replace=False)))
                # total_rand_index = torch.tensor(np.append(rand_index, [6,10,14,15,21,22]))
                total_rand_index = torch.tensor([6,10,14,15,16,17,21,22,23,24])
                z[:,total_rand_index,:] = torch.zeros(z[:,total_rand_index,:].size(), device="cuda")

            
        keep_index = [item for item in np.array(range(self.FLAGS.n_joints)) if item not in total_rand_index]
        
        z = self.encoder(z)

        if self.FLAGS.use_backbone:
            x = self._forward_features(x)
            x = x.unsqueeze(1)
            z = torch.cat((z, x), 1)

        if self.FLAGS.use_image_patches:
            x = self.patch_embed(x)
            x += self.pos_enc_image
            z += self.pos_enc
            z = torch.cat((z,x),1)

        # z += self.pos_enc

        z = self.transformer(z)

        z = self.decoder(z)

        return z[:, :self.FLAGS.n_joints, :], total_rand_index, keep_index

    def training_step(self, batch, batch_idx):
        labels = batch
        joints = labels['msk']
        images = labels['image']
        imgname = labels['imgname']
        ids = labels['id']

        joints_to_plot = rearrange(joints[0].unsqueeze(0).detach().cpu().numpy(),'b j d -> b (d j)')
        joints_denorm = denormalize_head2d(joints_to_plot,mode = int(ids[0].item()))
        joints_denorm = rearrange(joints_denorm, 'b (d j) -> b j d', d=2)

        # plot_wandb(imgname[0],joints_denorm[0],joints_denorm[0],ids[0].detach().cpu().numpy(),mode='save')

        z = joints.clone()

        if self.FLAGS.use_backbone or self.FLAGS.use_image_patches:
            x = images.clone()
        else:
            x = None


        # mapping = None
        # self.mapping_train = None

        if int(ids[0].item()) == 0:
            self.mapping_train = None
            mapping = None
        else:
            mapping = [ele for ele in self.body19_mapping if ele not in [0]]
            self.mapping_train = self.body19_mapping
        
        self.generated, self.mask_index, self.no_mask_index = self(z, x,mapping)
        self.total_generated = torch.empty(joints.size()).bfloat16().cuda()

        self.total_generated[:,self.mask_index,:] = self.generated[:,self.mask_index,:]
        self.total_generated[:,self.no_mask_index,:] = joints[:,self.no_mask_index,:].bfloat16()

        # self.total_generated = rearrange(self.total_generated, 'b j d -> b (d j)')

        ## For Plotting on Wandb

        self.joints_train = joints.clone()
        self.imgname_train = imgname
        self.predicion_train = self.total_generated
        self.id_train = ids

        ######

        if mapping != None:
            mask = torch.ones_like(self.mask_index, dtype=torch.bool)
            for element in self.remove_joints:
                mask = mask & (self.mask_index != element)
            mask_index_new = self.mask_index[mask]
        else:
            mask_index_new = self.mask_index

        joints_mask = joints[:,mask_index_new,:]
        joints_mask = rearrange(joints_mask, 'b j d -> b (d j)')

        joints_gen = self.generated[:,mask_index_new,:]
        joints_gen = rearrange(joints_gen, 'b j d -> b (d j)')

        mse_loss = self.mse_loss(joints_gen,joints_mask)
        self.log("Train MSE loss", mse_loss, prog_bar=True)

        return mse_loss
    
    def on_train_epoch_end(self):

        train_index = random.choice(range(self.FLAGS.batch_size))

        joints_predicted_train = rearrange(self.predicion_train[train_index].unsqueeze(0).float().detach().cpu().numpy(),'b j d -> b (d j)')
        joints_predicted_denorm_train = denormalize_head2d(joints_predicted_train,mode = int(self.id_train[0].item()))

        if self.mapping_train == None:
            joints_predicted_denorm_train = rearrange(joints_predicted_denorm_train, 'b (d j) -> b j d', d=2)
        else:
            joints_predicted_denorm_train = rearrange(joints_predicted_denorm_train, 'b (d j) -> b j d', d=2)[:,self.mapping_train,:]

        joints_gt_train = rearrange(self.joints_train[train_index].unsqueeze(0).float().detach().cpu().numpy(),'b j d -> b (d j)')
        joints_gt_denorm_train = denormalize_head2d(joints_gt_train,mode = int(self.id_train[train_index].item()))

        if self.mapping_train == None:
            joints_gt_denorm_train = rearrange(joints_gt_denorm_train, 'b (d j) -> b j d', d=2)
        else:
            joints_gt_denorm_train = rearrange(joints_gt_denorm_train, 'b (d j) -> b j d', d=2)[:,self.mapping_train,:]

        fig_train = plot_wandb(self.imgname_train[train_index],joints_gt_denorm_train[0],joints_predicted_denorm_train[0],self.id_train[train_index].detach().cpu().numpy(),mode='wandb')
        wandb.log({ 'Training' : wandb.Image(fig_train)})

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):

        labels = batch
        joints = labels['msk']
        images = labels['image']
        imgname = labels['imgname']
        ids = labels['id']

        z = joints.clone()
        
        if self.FLAGS.use_backbone or self.FLAGS.use_image_patches:
            x = images.clone()
        else:
            x = None

        # mapping = None
        # self.mapping_val = None

        if int(ids[0].item()) == 0:
            self.mapping_val = None
            mapping = None
        else:
            mapping = [ele for ele in self.body19_mapping if ele not in [0]]
            self.mapping_val = self.body19_mapping
        
        self.generated, self.mask_index, self.no_mask_index = self(z, x,mapping)
        self.total_generated = torch.empty(joints.size()).bfloat16().cuda()

        self.total_generated[:,self.mask_index,:] = self.generated[:,self.mask_index,:]
        self.total_generated[:,self.no_mask_index,:] = joints[:,self.no_mask_index,:].bfloat16()

        # self.total_generated = rearrange(self.total_generated, 'b j d -> b (d j)')

        ## For Plotting on Wandb

        self.joints_val = joints.clone()
        self.imgname_val = imgname
        self.predicion_val = self.total_generated
        self.id_val = ids

        if mapping != None:
            mask = torch.ones_like(self.mask_index, dtype=torch.bool)
            for element in self.remove_joints:
                mask = mask & (self.mask_index != element)
            mask_index_new = self.mask_index[mask]
        else:
            mask_index_new = self.mask_index

        joints_mask = joints[:,mask_index_new,:]
        joints_mask = rearrange(joints_mask, 'b j d -> b (d j)')

        joints_gen = self.generated[:,mask_index_new,:]
        joints_gen = rearrange(joints_gen, 'b j d -> b (d j)')

        mse_loss = self.mse_loss(joints_gen,joints_mask)
        self.log("Validation MSE loss", mse_loss, prog_bar=True)

        return mse_loss
    

    def on_validation_epoch_end(self):

        val_index = random.choice(range(self.FLAGS.batch_size))
        
        joints_predicted = rearrange(self.predicion_val[val_index].unsqueeze(0).float().detach().cpu().numpy(),'b j d -> b (d j)')
        joints_predicted_denorm = denormalize_head2d(joints_predicted,mode = int(self.id_val[val_index].item()))
        if self.mapping_val == None:
            joints_predicted_denorm = rearrange(joints_predicted_denorm, 'b (d j) -> b j d', d=2)
        else:
            joints_predicted_denorm = rearrange(joints_predicted_denorm, 'b (d j) -> b j d', d=2)[:,self.mapping_val,:]

        joints_gt = rearrange(self.joints_val[val_index].unsqueeze(0).float().detach().cpu().numpy(),'b j d -> b (d j)')
        joints_gt_denorm = denormalize_head2d(joints_gt,mode = int(self.id_val[val_index].item()))
        if self.mapping_val == None:
            joints_gt_denorm = rearrange(joints_gt_denorm, 'b (d j) -> b j d', d=2)
        else:
            joints_gt_denorm = rearrange(joints_gt_denorm, 'b (d j) -> b j d', d=2)[:,self.mapping_val,:]

        fig = plot_wandb(self.imgname_val[val_index],joints_gt_denorm[0],joints_predicted_denorm[0],self.id_val[val_index].detach().cpu().numpy(),mode='wandb')
        wandb.log({ 'Validation' : wandb.Image(fig) })

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        mse_loss = nn.MSELoss()
        labels = batch
        joints = labels['poses']
        images = labels['image']
        visibility = labels.get('vis', None)

        z = joints.clone()
        x = images.clone()

        self.generated, self.mask_index, self.no_mask_index = self(z, x, self.encoder,self.FLAGS, visibility_flags=visibility)

        self.total_generated = torch.empty(joints.size()).bfloat16().cuda()

        if not torch.is_tensor(self.mask_index):
            for inx in range(self.generated.shape[0]):
                self.total_generated[inx,self.mask_index[inx],:] = self.generated[inx,self.mask_index[inx],:]
                self.total_generated[inx,self.no_mask_index[inx],:] = joints[inx,self.no_mask_index[inx],:].bfloat16()
        else:
            self.total_generated[:,self.mask_index,:] = self.generated[:,self.mask_index,:]
            self.total_generated[:,self.no_mask_index,:] = joints[:,self.no_mask_index,:].bfloat16()
        
        # self.total_generated = rearrange(self.total_generated, 'b j d -> b (d j)')
        joints_mask = []
        joints_gen = []
        if not torch.is_tensor(self.mask_index):
            for inx in range(self.generated.shape[0]):
                # no need to intersect bc demo choice algo does not take zeroes
                joints_mask.append(joints[inx,self.mask_index[inx],:].flatten())
                joints_gen.append(self.generated[inx,self.mask_index[inx],:].flatten())
            joints_mask = torch.hstack(joints_mask)
            joints_gen = torch.hstack(joints_gen)
        else:
            for inx in range(joints.shape[0]):
                present_joints = (batch['vis'][inx] > 0).nonzero(as_tuple=False).flatten()
                true_mask = np.intersect1d(present_joints.cpu(), self.mask_index.cpu())
                joints_mask.append(joints[inx,true_mask,:].flatten())
                joints_gen.append(self.generated[inx,true_mask,:].flatten())
            joints_mask = torch.hstack(joints_mask)         
            joints_gen = torch.hstack(joints_gen)

        local_mse_loss = mse_loss(joints_gen,joints_mask)
        self.log("Test MSE loss", local_mse_loss, prog_bar=True)

        self.log_joints(joints[0], self.total_generated[0].float(), vis_flags=batch['vis'][0])
        
        # in the image
        original_joints = torch.zeros(joints.shape)
        generated_joints = torch.zeros(joints.shape)
        for inx in range(joints.shape[0]):
            # denorm
            if self.FLAGS.norm == 'head-pelvis':
                original_joints[inx] = denormalize_head(joints[inx], batch['scale_mean'][inx], batch['offset'][inx])
                generated_joints[inx] = denormalize_head(self.total_generated[inx].float(), batch['scale_mean'][inx], batch['offset'][inx])
                
                x_scale = batch['scales'][inx][0].cpu()
                y_scale = batch['scales'][inx][1].cpu()

                original_joints[inx][:, 0] = original_joints[inx][:, 0] / x_scale * 224
                original_joints[inx][:, 1] = original_joints[inx][:, 1] / y_scale * 224

                generated_joints[inx][:, 0] = generated_joints[inx][:, 0] / x_scale * 224
                generated_joints[inx][:, 1] = generated_joints[inx][:, 1] / y_scale * 224

            elif self.FLAGS.norm == 'pixel':
                # x_scale = batch['scales'][inx][0]
                # y_scale = batch['scales'][inx][1]

                # original_joints[inx][:, 0] = joints[inx][:, 0] * x_scale
                # original_joints[inx][:, 1] = joints[inx][:, 1] * y_scale

                # generated_joints[inx][:, 0] = self.total_generated[inx].float()[:, 0] * x_scale
                # generated_joints[inx][:, 1] = self.total_generated[inx].float()[:, 1] * y_scale
                original_joints = joints * 224
                generated_joints = self.total_generated.float() * 224

            
            # plot example in the image(to not repeat denorm outside of the cycle)
            if inx == 0:
                img = read_image(batch['image_path'][inx])
                img = torchvision.transforms.functional.resize(img, (224, 224)).cpu().numpy()
                img = np.moveaxis(img, 0, -1)

                self.log_joints(original_joints[0], generated_joints[0], vis_flags=batch['vis'][0], name=f"Pose viz denorm", image=img)
        
        joints_mask = []
        joints_gen = []
        if not torch.is_tensor(self.mask_index):
            # each example has separate mask index
            for inx in range(original_joints.shape[0]):
                joints_gen.append(generated_joints[inx,self.mask_index[inx].cpu(),:].flatten())
                joints_mask.append(original_joints[inx,self.mask_index[inx].cpu(),:].flatten())
            generated_joints = torch.hstack(joints_gen)
            original_joints = torch.hstack(joints_mask)
        else:
            # one mask index for all
            original_joints = original_joints[:,self.mask_index,:]
            original_joints = rearrange(original_joints, 'b j d -> b (d j)')

            generated_joints = generated_joints[:,self.mask_index,:]
            generated_joints = rearrange(generated_joints, 'b j d -> b (d j)')

        pixel_mse_loss = mse_loss(original_joints, generated_joints)
        self.log("Test pixel MSE loss", pixel_mse_loss, prog_bar=True)

    def log_joints(self, joints, generated, vis_flags=None, name='Pose viz', style='ro', anno=False, image=None, save='wandb', path=None):
        fig, ax = plt.subplots(1, 1)
        ax.invert_yaxis()

        if not image is None:
            ax.imshow(image)
        plot_pose(ax, joints, vis=vis_flags, annotate=anno)
        plot_pose(ax, generated, vis=vis_flags, style=style, annotate=anno)
        if save == 'wandb':
            wandb.log({f"{name}": wandb.Image(fig)})
        elif save == 'local':
            if path:
                save_path = os.path.join('plot_logs', path)
            else:
                save_path = os.path.join('plot_logs', 'common', f'{self.img_num}.png')
            plt.savefig(save_path)
        plt.close(fig)
    
    def configure_optimizers(self):
        lr = self.FLAGS.lr

        if self.FLAGS.use_backbone:
            fe_params = [
                x for x in self.feature_extractor.parameters()
            ]
            main_params = [
                p
                for p in self.parameters()
                if all(p is not x for x in self.feature_extractor.parameters())
            ]

        if self.FLAGS.use_backbone and self.FLAGS.freeze_fe:
            model_params = self.parameters()
        elif self.FLAGS.use_backbone and not self.FLAGS.freeze_fe:
            model_params = [{'params': main_params},
                            {'params': fe_params, 'lr': lr/10}
                            ]
        else:
            model_params = self.parameters()
        
    
        opt_encoder = torch.optim.Adam(model_params, lr=lr)
        scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_encoder, gamma=0.95)

        return [opt_encoder], [scheduler_encoder]