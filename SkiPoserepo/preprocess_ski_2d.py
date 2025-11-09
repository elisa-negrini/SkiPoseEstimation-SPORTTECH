import sys
import os
import json
import cv2
sys.path.append("..")
from utils import translation

'''
pose has this structure

joints = [
  'head', 'neck',
  'shoulder_right', 'elbow_right', 'hand_right', 'pole_basket_right',
  'shoulder_left', 'elbow_left', 'hand_left', 'pole_basket_left',
  'hip_right', 'knee_right', 'ankle_right',
  'hip_left', 'knee_left', 'ankle_left',
  'ski_tip_right', 'toes_right', 'heel_right', 'ski_tail_right',
  'ski_tip_left', 'toes_left', 'heel_left', 'ski_tail_left'
]

'''

annotation_path = "/home/federico.diprima/Ski_2DPose_Dataset/annotations"

with open(annotation_path+"/ski2dpose_labels.json") as f:
   data = json.load(f)

processed_data = []
for folder in data.keys():
    for subject in data[folder].keys():
        for image in data[folder][subject].keys():
            processed_data.append([data[folder][subject][image]["annotation"][idx][:2] for idx in range(24)])

# remove right and left pole basket from the dataset
idx_left_pole_basket = 5
idx_right_pole_basket = 9

for pose in processed_data:
    pose.pop(idx_right_pole_basket)
    pose.pop(idx_left_pole_basket)

'''
new pose has this structure

joints = [
  'head', 'neck',
  'shoulder_right', 'elbow_right', 'hand_right',
  'shoulder_left', 'elbow_left', 'hand_left',
  'hip_right', 'knee_right', 'ankle_right',
  'hip_left', 'knee_left', 'ankle_left',
  'ski_tip_right', 'toes_right', 'heel_right', 'ski_tail_right',
  'ski_tip_left', 'toes_left', 'heel_left', 'ski_tail_left'
]

'''

# add pelvis (center hip) to the pose --> (hip_right + hip_left) / 2
for pose in processed_data:
    pelvis = []
    pelvis.append((pose[8][0] + pose[11][0]) / 2)
    pelvis.append((pose[8][1] + pose[11][1]) / 2)
    pose.append(pelvis)

'''
new pose has this structure

joints = [
  'head', 'neck',
  'shoulder_right', 'elbow_right', 'hand_right',
  'shoulder_left', 'elbow_left', 'hand_left',
  'hip_right', 'knee_right', 'ankle_right',
  'hip_left', 'knee_left', 'ankle_left',
  'ski_tip_right', 'toes_right', 'heel_right', 'ski_tail_right',
  'ski_tip_left', 'toes_left', 'heel_left', 'ski_tail_left', 'center_hip'
]

'''

# train and validation split
train_data = processed_data

# save the files in json format
with open(annotation_path+'/ski2dpose_processed.json', 'w') as f:
    json.dump(train_data, f)


                                                
