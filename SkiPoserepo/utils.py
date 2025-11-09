from cProfile import label
from multiprocessing import connection
import torch
import numpy as np
from torch.utils.data import Dataset
import wandb
import os, glob
import re
import cv2
from random import shuffle
import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib.gridspec import GridSpec
from pydoc import classname
import torch.nn as nn

def connect2d(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    return x1,x2,y1,y2

def x_rotation(theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]],dtype=np.float)
    return R

def y_rotation(theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]],dtype=np.float)
    return R

def z_rotation(theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]],dtype=np.float)
    return R

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def plot_wandb(image_path,gt,pred,ids, mode = 'wandb'):
    plt.style.use('ggplot')
    img = {}
    fig = plt.figure(figsize=(30,10))
    # fig = plt.figure(figsize=(15,5),dpi=300)
    ax = fig.add_subplot(1,3,1)
    ax1 = fig.add_subplot(1,3,2)
    ax2 = fig.add_subplot(1,3,3)

    # Plot Image
    image = cv2.imread(image_path)
    ax.imshow(image)
    ax1.imshow(image)
    ax2.imshow(image)

    if ids == 0:
        bones = [[0,1],[1,2],[1,3],[3,4],[4,5],[5,6],[1,7],[7,8],[8,9],[9,10],[0,11],[11,12],[12,13],[13,14],[13,15],[14,15],[16,14],[15,17],[0,18],[18,19],[19,20],[20,21],[20,22],[21,22],[23,21],[22,24]]
        labels = ['pelvis','neck','head','r_shoulder','r_elbow','r_wrist','r_pole','l_shoulder','l_elbow','l_wrist','l_pole',
                  'r_hips','r_knee','r_ankle','r_toes','r_heel','r_ski_tip','r_ski_tail','l_hips','l_knee','l_ankle','l_toes','l_heel','l_ski_tip','l_ski_tail']
    else:
        bones = [[0,1],[1,2],[1,3],[3,4],[4,5],[1,6],[6,7],[7,8],[0,9],[9,10],[10,11],[12,13],[0,14],[14,15],[15,16],[17,18]]
        labels = ['pelvis','neck','head','r_shoulder','r_elbow','r_wrist','l_shoulder','l_elbow','l_wrist',
                  'r_hips','r_knee','r_ankle','r_ski_tip','r_ski_tail','l_hips','l_knee','l_ankle','l_ski_tip','l_ski_tail']

    color_bones_gt = '-b'
    color_joints_gt = 'b'

    color_bones_pred = '-r'
    color_joints_pred = 'r'
    
    ## Left Plot
    for i,connections in enumerate(bones):
        x1,x2,y1,y2 = connect2d(gt[:,0], gt[:,1], connections[0],connections[1])
        ax.plot([x1,x2],[y1,y2],color_bones_gt)
    for i,l in enumerate(labels):
        ax.scatter(gt[i,0], gt[i,1],color=color_joints_gt)

    ## Middle Plot
    for i,connections in enumerate(bones):
        x1,x2,y1,y2 = connect2d(gt[:,0], gt[:,1], connections[0],connections[1])
        ax1.plot([x1,x2],[y1,y2],color_bones_gt)
    for i,l in enumerate(labels):
        ax1.scatter(gt[i,0], gt[i,1],color=color_joints_gt)

    for i,connections in enumerate(bones):
        x1,x2,y1,y2 = connect2d(pred[:,0], pred[:,1], connections[0],connections[1])
        ax1.plot([x1,x2],[y1,y2],color_bones_pred)
    for i,l in enumerate(labels):
        ax1.scatter(pred[i,0], pred[i,1],color=color_joints_pred)

    ## Right Plot
        
    for i,connections in enumerate(bones):
        x1,x2,y1,y2= connect2d(pred[:,0], pred[:,1], connections[0],connections[1])
        ax2.plot([x1,x2],[y1,y2],color_bones_pred)
    for i,l in enumerate(labels):
        ax2.scatter(pred[i,0], pred[i,1],color=color_joints_pred)


    ax.set_title('GT',fontsize=10)
    ax.set_xticks([]) 
    ax.set_yticks([]) 

    ax1.set_title('GT + Pred Pose',fontsize=10)

    ax1.set_xticks([]) 
    ax1.set_yticks([]) 

    ax2.set_title('Prediction',fontsize=10)
    ax2.set_xticks([]) 
    ax2.set_yticks([]) 

    if mode == 'wandb':
        return fig
    else:
        fig.show()


def MPJPE(skeleton1, skeleton2):
    difference=[]

    if skeleton1.shape == skeleton2.shape:
        for i,joint in enumerate(skeleton1):
            difference.append(np.linalg.norm(skeleton1[i] - skeleton2[i]))

    res=0
    for x in difference:
        res+=x

    return res/len(skeleton1)

def normalize_head2d(poses_2d, root_joint=0):
    # center at root joint
    p2d = poses_2d.reshape(-1, 2, 25)

    # p2d = poses_2d.reshape(-1, 2, 19)
    # p2d -= p2d[:, :, [root_joint]]
    scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 2], axis=1, keepdims=True)
    p2ds = p2d / scale.mean()
    p2ds = p2ds * (1 / 10)
    p2ds = p2ds.reshape(-1,50)
    # p2ds = p2ds.reshape(-1,38)
    return p2ds, scale.mean()

def denormalize_head2d(poses_2d, mode):
    # center at root joint
    if mode == 1:
        scale = 44.335396
    else:
        scale = 63.86765

    p2d = poses_2d.reshape(-1, 2, 25)

    # p2d = poses_2d.reshape(-1, 2, 19)
    # p2d -= p2d[:, :, [root_joint]]
    # scale = np.linalg.norm(p2d[:, :, 0] - p2d[:, :, 2], axis=1, keepdims=True)
    p2ds = p2d * scale
    p2ds = p2ds / (1 / 10)
    p2ds = p2ds.reshape(-1,50)

    # p2ds = p2ds.reshape(-1,38)
    return p2ds


def flip_joints_np(joints):
    # Flip the annotations
    flipped_annotations = []
    
    for joint in joints:
        # Flip the x-coordinate
        flipped_x = 224 - joint[0] - 1
        # Keep the y-coordinate as is
        flipped_y = joint[1]
        flipped_annotations.append([flipped_x, flipped_y])

    flipped_joints = np.empty(shape=joints.shape)

    flipped_joints = np.array(flipped_annotations)

    return flipped_joints

def flip_joints(joints, flip_pairs, dimension=2): #"b (d j) -> b j d"
    # joints = rearrange(joints, "(d j) -> j d", d = dimension)

    for i in range(joints.shape[0]):
        joints[i, 0] = -joints[i, 0]

    joints_flipped = joints.clone()
    
    for left, right in flip_pairs:
        joints_flipped[left] = joints[right]
        joints_flipped[right] = joints[left]

    # joints_flipped = rearrange(joints_flipped, "j d -> (d j)")

    return joints_flipped


def get_bone_lengths_all(poses, n_joints=17, bone_map=None,type='torch'):
    
    if bone_map == None:
        bone_map = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],
                            [16,18],[17,19],[18,20],[19,21],[20,22],[21,23]]

    poses = poses.reshape((-1, 3, n_joints))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    if type == 'torch':
        bone_lengths = torch.norm(bones, p=2, dim=1)
    else:
        bone_lengths = np.linalg.norm(bones, ord=2, axis=1)

    return bone_lengths

def load_model(pose2rot,modelFolder,numBetas):
    
    # model_male = smplx.create(modelFolder, model_type='smpl',
    #                             gender='male',
    #                             ext='npz')
    # model_female = smplx.create(modelFolder, model_type='smpl',
    #                             gender='female',
    #                             ext='npz')
    model_neutral = smplx.create(modelFolder, model_type='smpl',
                                    gender='neutral',
                                    ext='pkl')
    return model_neutral

