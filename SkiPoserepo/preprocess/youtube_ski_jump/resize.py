import cv2
import os
import pickle as pkl
import numpy as np


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

image_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images_cropped/Validation'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images_resized/Validation'
input_folder_poses = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/skel_cropped/val.pkl'
output_folder_poses = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/skel_resized/val.pkl'

# pickle_off = open(output_folder_poses, "rb")
# loaddata = pkl.load(pickle_off)

# print(len(loaddata['gt_joints']))

# quit()

# Corresponding joints
joints = ['head','neck','rsho','relb','rhan','lsho','lelb','lhan','hip','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']

# Bones for drawing examples
bones =[[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,15],[11,16],[9,12],[12,13],[13,14],[14,17],[14,18]]


pickle_off = open(input_folder_poses, "rb")
loaddata = pkl.load(pickle_off)

len_dataset = len(os.listdir(image_folder))

data = {}
poses = np.empty(shape=(len_dataset,len(joints),3))


def resize_images(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, img in enumerate(os.listdir(image_folder)):
        print(img)
        img_path = os.path.join(image_folder,img)
        image = cv2.imread(img_path)

        
        if image is not None:

            skel = loaddata['gt_joints'][i]
            resized_image = cv2.resize(image, (224, 224))

            skel_resize = np.empty(shape=skel.shape)
            skel_resize[:,0] = skel[:,0] * 224 / image.shape[1]
            skel_resize[:,1] = skel[:,1] * 224 / image.shape[0]
            skel_resize[:,2] = skel[:,2]

            # cropped_image_test = annotate_img(resized_image, skel_resize[:,:-1],skel_resize[:,2])

            # cv2.imshow("test cropped",cropped_image_test)
            # cv2.waitKey(0)

            poses[i,:,:] = skel_resize

            output_path = os.path.join(output_folder, img)
            # cv2.imwrite(output_path, resized_image)

            print(f"Resized {img} and saved to {output_path}")


    data['gt_joints'] = poses

    with open(output_folder_poses, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

resize_images(image_folder,output_folder)