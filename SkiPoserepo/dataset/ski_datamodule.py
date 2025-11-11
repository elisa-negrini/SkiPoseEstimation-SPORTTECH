from __future__ import division
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import numpy as np
import cv2
import pickle as pkl
import pytorch_lightning as pl
from torchvision import transforms
import os
from einops import rearrange
from utils import normalize_head2d, flip_joints_np, plot_wandb,denormalize_head2d
from torchvision.io import read_image
import cv2
from dataset.mask_generator import MaskGenerator


class SkiDatamodule(pl.LightningDataModule):

    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    def train_dataloader(self):
        
        train_dataset = Ski_Dataset(self.FLAGS,'train.pkl','Training')

        train_loader =  DataLoader(train_dataset,shuffle=False,pin_memory=True,num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)
        
        return train_loader

    def val_dataloader(self):

        val_dataset = Ski_Dataset(self.FLAGS,'val.pkl','Validation')

        val_loader =  DataLoader(val_dataset,shuffle=False,pin_memory=True,num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)
        
        return val_loader  

    def test_dataloader(self):

        test_dataset = Ski_Dataset(self.FLAGS,'val.pkl','Validation')

        test_loader =  DataLoader(test_dataset,shuffle=False,pin_memory=True,num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return test_loader    

class Ski_Dataset(Dataset):

    def __init__(self, FLAGS, skel_name, img_folder, nomalize_func = normalize_head2d ):

        self.FLAGS = FLAGS

        ## Load Ski2D Dataset
        skel_path = os.path.join(self.FLAGS.data_path,'Ski_2DPose_Dataset','skel_resized', skel_name)
        img_path = os.path.join(self.FLAGS.data_path,'Ski_2DPose_Dataset','images_resized', img_folder)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # self.mask_generator = MaskGenerator(
        #     input_size=self.FLAGS.img_size,
        #     mask_patch_size=self.FLAGS.mask_size,
        #     mask_ratio=self.FLAGS.mask_ratio,
        # )

        # print('Shape Data ',loaddata['gt_joints'].shape)

        self.data = dict()

        mapping = [2,1,7,8,9,10,3,4,5,6,0,18,19,20,11,12,13,23,21,22,24,16,14,15,17]
        joints_to_keep = [0,1,2,3,4,5,7,8,9,11,12,13,16,17,18,19,20,23,24]

        # Read Image Paths

        img_paths_ski2d_normal = np.array([os.path.join(img_path,im) for im in os.listdir(img_path)])

        # Save Dataset FLAGS

        flags_dataset_ski2d_normal = torch.zeros(size=(len(img_paths_ski2d_normal),1))
        
        # Read 2D Pose Data

        pickle_off = open(skel_path, "rb")
        loaddata = pkl.load(pickle_off)

        ski2d_poses2d_normal = np.zeros(shape=(loaddata['gt_joints'].shape[0],25,2))
        ski2d_poses2d_normal[:,mapping,:] = loaddata['gt_joints'][:,:,:2]

        # ski2d_poses2d_normal = ski2d_poses2d_normal[:,joints_to_keep,:]

        # ski2d_poses2d_normal = rearrange(ski2d_poses2d_normal,'b j d -> b (d j)')
        
        # ski2d_poses2d_normal = torch.tensor(ski2d_poses2d_normal, dtype=torch.float)
        # ski2d_poses2d_normal, scale_ski2d = nomalize_func(ski2d_poses2d_normal)
        # ski2d_poses2d_normal = rearrange(ski2d_poses2d_normal, 'b (d j) -> b j d', d=2)

        ## Add Data by flipping

        img_path = os.path.join(self.FLAGS.data_path,'Ski_2DPose_Dataset','images_flipped', img_folder)

        img_paths_ski2d_flipped = np.array([os.path.join(img_path,im) for im in os.listdir(img_path)])

        flags_dataset_ski2d_flipped = torch.zeros(size=(len(img_paths_ski2d_flipped),1))

        ski2d_poses2d_flipped = np.empty(shape=ski2d_poses2d_normal.shape)

        for i in range(ski2d_poses2d_normal.shape[0]):
            ski2d_poses2d_flipped[i,:,:] = flip_joints_np(ski2d_poses2d_normal[i,:,:])


        ski2d_poses2d_normal = rearrange(ski2d_poses2d_normal,'b j d -> b (d j)')
        ski2d_poses2d_normal = torch.tensor(ski2d_poses2d_normal, dtype=torch.float)
        ski2d_poses2d_normal, scale_ski2d = nomalize_func(ski2d_poses2d_normal)
        ski2d_poses2d_normal = rearrange(ski2d_poses2d_normal, 'b (d j) -> b j d', d=2)

        ski2d_poses2d_flipped = rearrange(ski2d_poses2d_flipped,'b j d -> b (d j)')
        ski2d_poses2d_flipped = torch.tensor(ski2d_poses2d_flipped, dtype=torch.float)
        ski2d_poses2d_flipped, scale_ski2d = nomalize_func(ski2d_poses2d_flipped)
        ski2d_poses2d_flipped = rearrange(ski2d_poses2d_flipped, 'b (d j) -> b j d', d=2)

        ## Concatenate Normal and Flipped

        ski2d_poses2d = torch.cat((ski2d_poses2d_normal,ski2d_poses2d_flipped))
        img_paths_ski2d = np.concatenate((img_paths_ski2d_normal,img_paths_ski2d_flipped))
        flags_dataset_ski2d = torch.cat((flags_dataset_ski2d_normal,flags_dataset_ski2d_flipped))


        ## Load YTB Dataset

        self.FLAGS = FLAGS
        skel_path = os.path.join(self.FLAGS.data_path,'youtube_skijump_dataset','skel_resized', skel_name)
        img_path = os.path.join(self.FLAGS.data_path,'youtube_skijump_dataset','images_resized', img_folder)

        # print('Shape Data ',loaddata['gt_joints'].shape)
        mapping = [2,1,3,4,5,7,8,9,0,11,12,13,18,19,20,16,17,23,24]

        to_keep = [0,1,2,3,4,5,7,8,9,11,12,13,16,17,18,19,20,23,24]

        # Read Image Paths

        img_paths_ytb_ski_normal = np.array([os.path.join(img_path,im) for im in os.listdir(img_path)])

        # Save Dataset FLAGS

        flags_dataset_ytb_ski_normal = torch.ones(size=(len(img_paths_ytb_ski_normal),1))
        
        # Read 2D Pose Data

        pickle_off = open(skel_path, "rb")
        loaddata = pkl.load(pickle_off)

        ytb_ski_poses2d_normal = np.zeros(shape=(loaddata['gt_joints'].shape[0],25,2))

        ytb_ski_poses2d_normal[:,mapping,:] = loaddata['gt_joints'][:,:,:2]


        # ytb_ski_poses2d_normal = ytb_ski_poses2d_normal[:,joints_to_keep,:]

        # ytb_ski_poses2d_normal = rearrange(ytb_ski_poses2d_normal,'b j d -> b (d j)')
        
        
        # ytb_ski_poses2d_normal = torch.tensor(ytb_ski_poses2d_normal, dtype=torch.float)
        # ytb_ski_poses2d_normal, scale_ytb = nomalize_func(ytb_ski_poses2d_normal)
        # ytb_ski_poses2d_normal = rearrange(ytb_ski_poses2d_normal, 'b (d j) -> b j d', d=2)

        ## Add Data by flipping

        img_path = os.path.join(self.FLAGS.data_path,'youtube_skijump_dataset','images_flipped', img_folder)

        img_paths_ytb_ski_flipped = np.array([os.path.join(img_path,im) for im in os.listdir(img_path)])

        flags_dataset_ytb_ski_flipped = torch.ones(size=(len(img_paths_ytb_ski_flipped),1))

        ytb_ski_poses2d_flipped = np.zeros(shape=ytb_ski_poses2d_normal.shape)

        for i in range(ytb_ski_poses2d_normal.shape[0]):
            ytb_ski_poses2d_flipped[i,:,:] = flip_joints_np(ytb_ski_poses2d_normal[i,:,:])


        ytb_ski_poses2d_normal = rearrange(ytb_ski_poses2d_normal,'b j d -> b (d j)')
        ytb_ski_poses2d_normal = torch.tensor(ytb_ski_poses2d_normal, dtype=torch.float)
        ytb_ski_poses2d_normal, scale_ytb = nomalize_func(ytb_ski_poses2d_normal)
        ytb_ski_poses2d_normal = rearrange(ytb_ski_poses2d_normal, 'b (d j) -> b j d', d=2)


        ytb_ski_poses2d_flipped = rearrange(ytb_ski_poses2d_flipped,'b j d -> b (d j)')
        ytb_ski_poses2d_flipped = torch.tensor(ytb_ski_poses2d_flipped, dtype=torch.float)
        ytb_ski_poses2d_flipped, scale_ytb = nomalize_func(ytb_ski_poses2d_flipped)
        ytb_ski_poses2d_flipped = rearrange(ytb_ski_poses2d_flipped, 'b (d j) -> b j d', d=2)


        ## Concatenate Normal and Flipped

        ytb_ski_poses2d = torch.cat((ytb_ski_poses2d_normal,ytb_ski_poses2d_flipped))
        img_paths_ytb_ski = np.concatenate((img_paths_ytb_ski_normal,img_paths_ytb_ski_flipped))
        flags_dataset_ytb_ski = torch.cat((flags_dataset_ytb_ski_normal,flags_dataset_ytb_ski_flipped))


        

        ## Concatenate the 2 Datasets

        # ski2d_poses2d = ski2d_poses2d_normal
        # ytb_ski_poses2d = ytb_ski_poses2d_normal
        # img_paths_ski2d = img_paths_ski2d_normal
        # img_paths_ytb_ski = img_paths_ytb_ski_normal
        # flags_dataset_ski2d = flags_dataset_ski2d_normal
        # flags_dataset_ytb_ski = flags_dataset_ytb_ski_normal

        # mask_ski_2d = torch.empty(size=(len(img_paths_ski2d),56,56))
        # mask_ytb_ski = torch.empty(size=(len(img_paths_ytb_ski),56,56))

        # for i in range(len(img_paths_ski2d)):
        #     mask_ski_2d[i] = torch.tensor(self.mask_generator())

        # for i in range(len(img_paths_ytb_ski)):
        #     mask_ytb_ski[i] = torch.tensor(self.mask_generator())

        self.data['poses_2d'] = torch.cat((ski2d_poses2d,ytb_ski_poses2d))
        self.data['img_paths'] = np.concatenate((img_paths_ski2d,img_paths_ytb_ski))
        self.data['flags_dataset'] = torch.cat((flags_dataset_ski2d,flags_dataset_ytb_ski))

        # self.data['mask'] = torch.cat((mask_ski_2d,mask_ytb_ski))

        print('ok')

        # self.data['poses_2d'] = ski2d_poses2d
        # self.data['img_paths'] = img_paths_ski2d
        # self.data['flags_dataset'] = flags_dataset_ski2d
    

    def __len__(self):
        return self.data['poses_2d'].shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        if self.FLAGS.mode == 'train':
            # if(torch.randint(low=0, high=2, size=(1,))):
            #     sample['imgname'] = self.data['img_paths'][idx]
            #     image = cv2.imread(sample['imgname'])
            #     flip_pairs = [[3,7],[4,8],[5,9],[6,10],[11,18],[12,19],[13,20],[14,21],[15,22],[16,23],[17,24]]
            #     sample['msk'] = flip_joints(self.data['poses_2d'][idx], flip_pairs, dimension=2)
            #     image = cv2.flip(image, 1)
            #     sample['image'] = self.transform(image)
            #     sample['id'] = self.data['flags_dataset'][idx]
            # else:
            sample['msk'] = self.data['poses_2d'][idx]
            sample['imgname'] = self.data['img_paths'][idx]
            image = cv2.imread(sample['imgname'])
            sample['image'] = self.transform(image)
            sample['id'] = self.data['flags_dataset'][idx]
            # sample['mask'] = self.data['mask']

        return sample