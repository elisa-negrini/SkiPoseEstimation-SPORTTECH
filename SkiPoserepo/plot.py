import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from utils import translation, invert_translation, normalize_head
import json

def plot(pred_pose, gt_pose, mapping, result_path, n_epoch):

    # create the results directory
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    gtx, gty = zip(*gt_pose)
    predx, predy = zip(*pred_pose)

    fig, ax = plt.subplots()
    ax.plot(gtx, gty, 'go', markersize=10)
    ax.plot(predx, predy, 'ro', markersize=1)

    for connection in mapping:
        joint_1 = connection[0]
        joint_2 = connection[1]
        ax.plot([gtx[joint_1], gtx[joint_2]], [gty[joint_1], gty[joint_2]], 'go-')
        ax.plot([predx[joint_1], predx[joint_2]], [predy[joint_1], predy[joint_2]], 'ro-')

    ax.set_xlim([min(min(predx), min(gtx))-0.01, max(max(predx), max(gtx))+0.01])
    ax.set_ylim([max(max(predy), max(gty))+0.01, min(min(predy), min(gty))-0.01])

    ax.set_title(f"Epoch {n_epoch} ")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Salvataggio del grafico come immagine
    output_file = os.path.join(result_path, f"epoch{n_epoch}.png")
    plt.savefig(output_file)


def plot_cv2(pose, image, bone_list, save_dir, file_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img = cv2.imread(os.path.join("/home/federico.diprima/Ski_Jump_BODY25_poses/images",image))
    height = img.shape[0]
    width = img.shape[1]
    no_ski = False

    # deve essere maggiore di 1 probabilmente per problemi di approssimazione
    for joint in pose:
        if joint[0] >= 1 and joint[1] >= 1:
            cv2.circle(img, (int(joint[0]),int(joint[1])), radius=5, color=(0, 255, 0), thickness=-1)
    
    # no plot ski if out of images
    if no_ski:
        for limb in no_ski_mapping:
            #if pose[limb[0]][0] >= 1 and pose[limb[0]][1] >= 1 and pose[limb[1]][0] >= 1 and pose[limb[1]][1] >= 1:
            cv2.line(img, (int(pose[limb[0]][0]), int(pose[limb[0]][1])) ,(int(pose[limb[1]][0]), int(pose[limb[1]][1])) , color=(0, 255, 0), thickness=2) 
    else:
        for limb in bone_list:
            #if pose[limb[0]][0] >= 1 and pose[limb[0]][1] >= 1 and pose[limb[1]][0] >= 1 and pose[limb[1]][1] >= 1:
            cv2.line(img, (int(pose[limb[0]][0]), int(pose[limb[0]][1])) ,(int(pose[limb[1]][0]), int(pose[limb[1]][1])) , color=(0, 255, 0), thickness=2) 
    
    output_file_path = os.path.join(save_dir, file_name)
    cv2.imwrite(output_file_path, img)


def plot_body(pose, image, bone_list, save_dir, file_name):

    img = cv2.imread(os.path.join("/home/federico.diprima/Ski_Jump_BODY25_poses/images",image))
    height = img.shape[0]
    width = img.shape[1]

    # deve essere maggiore di 1 probabilmente per problemi di approssimazione
    for i,joint in enumerate(pose):
        if joint[0] >= 1 and joint[1] >= 1 and i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,22]:
            cv2.circle(img, (int(joint[0]),int(joint[1])), radius=5, color=(0, 255, 0), thickness=-1)
   
    for limb in bone_list:
        #if pose[limb[0]][0] >= 1 and pose[limb[0]][1] >= 1 and pose[limb[1]][0] >= 1 and pose[limb[1]][1] >= 1:
        if limb[0] in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,22] and limb[1] in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,22]:
            cv2.line(img, (int(pose[limb[0]][0]), int(pose[limb[0]][1])) ,(int(pose[limb[1]][0]), int(pose[limb[1]][1])) , color=(0, 255, 0), thickness=2) 
    
    output_file_path = os.path.join(save_dir, file_name)
    cv2.imwrite(output_file_path, img)
