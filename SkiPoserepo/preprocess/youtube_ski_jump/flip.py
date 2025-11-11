import os
import cv2

image_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images_resized/Validation'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images_flipped/Validation'
output_folder_poses = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/skel_resized/val.pkl'


import os
import cv2
import numpy as np
import pickle as pkl
import sys
sys.path.append("/home/giuliamartinelli/Code/SkiPose")
from utils import flip_joints_np

# Corresponding joints
joints = ['head','neck','rsho','relb','rhan','lsho','lelb','lhan','hip','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']

joints_name = ['head','rsho','relb','rhan','lsho','lelb','lhan','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']

# Bones for drawing examples
bones = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,15],[11,16],[9,12],[12,13],[13,14],[14,17],[14,18]]


def annotate_img(img, an, vis, info=None):
    '''
    Annotates a given image with all joints. Visible joints will be drawn
    with a red circle, while invisible ones with a blue one.

    Args:
        :img: Input image (cv2 image)
        :an: Annotation positions tensor (in pixels)
        :vis: Visibility flag tensor
        :info: (video_id, split_id, img_id, frame_idx) tuple
    '''
    height, width, _ = img.shape  # Assuming the image is in HxWxC format

    # Scale based on head-foot distance
    scale = np.linalg.norm(an[0] - an[15]) / height

    # Create a copy of the input image to avoid modifying the original image
    img_an = img.copy()

    # Draw all bones
    for bone_from, bone_to in bones:
        x_from, y_from = an[bone_from]
        x_to, y_to = an[bone_to]
        cv2.line(img_an, (int(x_from), int(y_from)), (int(x_to), int(y_to)), (0, 255, 0), max(2, int(5*scale)))

    # Draw all joints
    for (x, y), flag in zip(an, vis):
        color = (0, 0, 255) if flag == 1 else (255, 0, 0)
        cv2.circle(img_an, (int(x), int(y)), max(2, int(14*scale)), color, -1)

    # Draw image name and frame number if given
    if info is not None:
        text = 'Image {}, frame {}.'.format(info[2], info[3])
        cv2.putText(img_an, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * (width / 1920),
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img_an, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5 * (width / 1920),
                    (255, 255, 255), 2, cv2.LINE_AA)

    return img_an


pickle_off = open(output_folder_poses, "rb")
loaddata = pkl.load(pickle_off)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, img in enumerate(os.listdir(image_folder)):

    img_path = os.path.join(image_folder,img)
    image = cv2.imread(img_path)
    image_flipped = cv2.flip(image,1)

    joints = loaddata['gt_joints'][i]

    h,w, _ = image.shape

    # Flip the annotations
    flipped_annotations = []
    for joint in joints:
        # Flip the x-coordinate
        flipped_x = w - joint[0] - 1
        # Keep the y-coordinate as is
        flipped_y = joint[1]
        flipped_annotations.append([flipped_x, flipped_y])

    flipped_joints = np.empty(shape=joints.shape)

    flipped_joints[:,:-1] = np.array(flipped_annotations)
    flipped_joints[:,2] = joints[:,2]

    # flip_pairs = [[3,7],[4,8],[5,9],[6,10],[11,18],[12,19],[13,20],[14,21],[15,22],[16,23],[17,24]]
    # flipped_joints = flip_joints_np(joints, flip_pairs, dimension=2)

    image_plot = annotate_img(image_flipped,flipped_joints[:,:-1],flipped_joints[:,2])
    cv2.imshow("test",image_plot)
    cv2.waitKey(0)

    # output_path = os.path.join(output_folder, img)
    # cv2.imwrite(output_path, image_flipped)
    # print(f"Flipped {img} and saved to {output_path}")