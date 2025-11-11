import cv2
import os
import json
import numpy as np
import pickle as pkl

label_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/annotations/ski2dpose_labels.json'
data_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images/Training'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images_cropped/Training'
output_folder_poses = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/skel_cropped/train.pkl'


# pickle_off = open(output_folder_poses, "rb")
# loaddata = pkl.load(pickle_off)

# print(len(loaddata['gt_joints']))

# quit()

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

def compute_bounding_box(skeleton_coordinates, image_width, image_height):
    # Extract x and y coordinates from the skeleton
    x_coordinates = [point[0] for point in skeleton_coordinates]
    y_coordinates = [point[1] for point in skeleton_coordinates]

    # Compute the bounding box
    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    # Calculate additional width and height
    additional_width = (max_x - min_x) / 6
    additional_height = (max_y - min_y) / 6

    # Expand the bounding box, considering image boundaries
    expanded_min_x = max(0, min_x - additional_width)
    expanded_max_x = min(image_width, max_x + additional_width)
    expanded_min_y = max(0, min_y - additional_height)
    expanded_max_y = min(image_height, max_y + additional_height)

    # Return the computed bounding box
    bounding_box = {
        'min_x': expanded_min_x,
        'max_x': expanded_max_x,
        'min_y': expanded_min_y,
        'max_y': expanded_max_y
    }

    return bounding_box

def plot_bbox_on_image(image, bounding_box, color=(0, 255, 0), thickness=2):
    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the bounding box coordinates to integers
    min_x, max_x, min_y, max_y = map(int, (bounding_box['min_x'], bounding_box['max_x'], bounding_box['min_y'], bounding_box['max_y']))

    # Draw the bounding box on the image
    cv2.rectangle(image_rgb, (min_x, min_y), (max_x, max_y), color, thickness)

    # Convert back to BGR for displaying with OpenCV
    image_with_bbox = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_with_bbox

def make_square(bbox,center):
    x_min, y_min, width, height = bbox  # Assuming bbox is represented by (x1, y1, x2, y2)


    # Determine the size of the square (side length)
    side_length = max(width, height)


    # Calculate the center of the original bounding box
    center_x = center[0]
    center_y = center[1]

    # Calculate the new coordinates for the square bounding box
    new_x_min = int(center_x - side_length / 2)
    new_y_min = int(center_y - side_length / 2)
    new_width = int(side_length)
    new_height = int(side_length)

    return np.array([new_x_min, new_y_min, new_width, new_height])

def draw_bbox(image, bbox,center):
    """
    Draw a bounding box on an image using OpenCV.

    Parameters:
        image (numpy.ndarray): The input image.
        bbox (tuple): Bounding box coordinates in format (x_min, y_min, width, height).

    Returns:
        image_with_bbox (numpy.ndarray): Image with bounding box drawn.
    """
    # Make a copy of the image to avoid modifying the original image
    image_with_bbox = image.copy()

    # Extract bbox coordinates
    x_min, y_min, width, height = bbox

    # ul_corner = np.array([int((x_min - width / 2)),int((y_min - height / 2))])
    # # center = ul_corner + 0.5 * np.array([bbox_width,bbox_height])

    # center = np.array([x_min - 0.5*width,y_min - 0.5*height])

    cv2.circle(image_with_bbox, (int(center[0]), int(center[1])), 3, (255, 0, 0), -1)

    # Draw bounding box rectangle
    cv2.rectangle(image_with_bbox, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

    return image_with_bbox

# Load annotations
with open(label_path) as f:
    labels = json.load(f)

# Corresponding joints
joints = ['head', 'neck',
            'shoulder_right', 'elbow_right', 'hand_right', 'pole_basket_right',
            'shoulder_left', 'elbow_left', 'hand_left', 'pole_basket_left', 'pelvis',
            'hip_right', 'knee_right', 'ankle_right',
            'hip_left', 'knee_left', 'ankle_left',
            'ski_tip_right', 'toes_right', 'heel_right', 'ski_tail_right',
            'ski_tip_left', 'toes_left', 'heel_left', 'ski_tail_left']

# Bones for drawing examples
bones = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9], [1,10],
            [10,11], [11,12], [12,13], [10,14],[14,15], [15,16],
            [17,18], [18,19], [19,20], [13,18], [13,19],
            [21,22], [22,23], [23,24], [16,22], [16,23]]

# Check if all images exist and index them
index_list = []

len_dataset = len(os.listdir(data_path))

data = {}
poses = np.empty(shape=(len_dataset,len(joints),3))

for i, img in enumerate(os.listdir(data_path)):
    print(img)
    img_path = os.path.join(data_path,img)
    image = cv2.imread(img_path)
    h,w, _ = image.shape

    video_id = img.split('_')[0]

    if video_id == 'OeDt6D':
        video_id = 'OeDt6D_9Rh4'
        split_id = img.split(video_id + '_')[-1].split('_')[0]
        image_id = img.split('.')[0]
    elif video_id == 'SVYdH614G':
        video_id = 'SVYdH614G_M'
        split_id = img.split(video_id + '_')[-1].split('_')[0]
        image_id = img.split('.')[0]
    elif video_id == 'v7PADD2':
        video_id = 'v7PADD2_I00'
        split_id = img.split(video_id + '_')[-1].split('_')[0]
        image_id = img.split('.')[0]
    else:
        split_id = img.split('_')[1]
        image_id = img.split('.')[0]

    skel = np.array(labels[video_id][split_id][image_id]["annotation"])

    joint_coord_new = np.empty(shape=(25,3))

    hips = (skel[10,:] + skel[13,:]) / 2

    mapping = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

    joint_coord_new[mapping,:] = skel
    joint_coord_new[10,:] = hips

    skel_pix = np.empty(shape=joint_coord_new.shape)
    skel_pix[:,0] = joint_coord_new[:,0]*image.shape[1]
    skel_pix[:,1] = joint_coord_new[:,1]*image.shape[0]
    skel_pix[:,2] = joint_coord_new[:,2]

    bounding_box = compute_bounding_box(skel_pix[:,:-1], w, h)

    x_top_left = int(bounding_box['min_x'])
    y_top_left = int(bounding_box['min_y'])
    # Calculate the bottom-right corner coordinates for the bounding box
    x_bottom_right = int(bounding_box['max_x'])
    y_bottom_right = int(bounding_box['max_y'])

    bbox_width = x_bottom_right - x_top_left
    bbox_height = y_bottom_right - y_top_left    

    bbox = np.array([x_top_left,y_top_left,bbox_width,bbox_height])

    center_y = (y_top_left + y_bottom_right) / 2
    center_x = (x_top_left + x_bottom_right) / 2

    center = np.array([center_x,center_y])

    squared_bbox = make_square(bbox,center)


    x_top_left = squared_bbox[0]
    y_top_left = squared_bbox[1]

    # Calculate the bottom-right corner coordinates for the bounding box
    x_bottom_right = int((squared_bbox[0] + squared_bbox[2]))
    y_bottom_right = int((squared_bbox[1] + squared_bbox[3]))

    max_offset = 500

    image_new = np.full(shape=(h+max_offset,w+max_offset,3), fill_value = (0,0,0),dtype=np.uint8)

    image_new[max_offset:max_offset+h,max_offset:max_offset+w] = image
    cropped_image = image_new[max_offset + y_top_left:max_offset + y_bottom_right,max_offset + x_top_left:max_offset + x_bottom_right]

    # skel_pix_bbox = np.empty(shape=skel_pix.shape)
    # skel_pix_bbox[:,0] = skel_pix[:,0]-int(squared_bbox[0])
    # skel_pix_bbox[:,1] = skel_pix[:,1]-int(squared_bbox[1])
    # skel_pix_bbox[:,2] = skel_pix[:,2]

    # poses[i,:,:] = skel_pix_bbox

    # cropped_image_test = annotate_img(cropped_image, skel_pix_bbox[:,:-1],skel_pix_bbox[:,2])

    # cv2.imshow("cropped_image",cropped_image_test)
    # cv2.waitKey(0)

    # print('here ok')

    # bbox = np.array([])

    # image_plot = annotate_img(image, skel_pix[:,:-1],skel_pix[:,2])

    # image_plot = plot_bbox_on_image(image_plot, bounding_box)
    # cv2.imshow("test",image_plot)
    # cv2.waitKey(0)

    # cropped_image = image[int(bounding_box['min_y']):int(bounding_box['max_y']),int(bounding_box['min_x']):int(bounding_box['max_x'])]
    skel_pix_bbox = np.empty(shape=skel_pix.shape)
    skel_pix_bbox[:,0] = skel_pix[:,0]-int(squared_bbox[0])
    skel_pix_bbox[:,1] = skel_pix[:,1]-int(squared_bbox[1])
    skel_pix_bbox[:,2] = skel_pix[:,2]

    poses[i,:,:] = skel_pix_bbox
    
    # cropped_image_test = annotate_img(cropped_image, skel_pix_bbox[:,:-1],skel_pix_bbox[:,2])

    # cv2.imshow("test cropped",cropped_image_test)
    # cv2.waitKey(0)

    # print('here ok')
    
    # Save the cropped image to the output folder
    # output_path = os.path.join(output_folder, img)
    # cv2.imwrite(output_path, cropped_image)

    # print(f"Cropped {image_filename} and saved to {output_path}")

data['gt_joints'] = poses

with open(output_folder_poses, 'wb') as handle:
    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)



quit()

for video_id, all_splits in labels.items():
    for split_id, split in all_splits.items():
        for img_id, img_labels in split.items():
            img_path = os.path.join(imgs_dir, video_id, split_id, '{}.{}'.format(img_id, img_extension))
            if os.path.exists(img_path):
                if ((mode == 'all') or
                    (mode == 'train' and (video_id, split_id) not in val_splits) or
                    (mode == 'val' and (video_id, split_id) in val_splits)):
                    index_list.append((video_id, split_id, img_id))
            else:
                print('Did not find image {}/{}/{}.{}'.format(video_id, split_id, img_id, img_extension))

quit()
def crop_images_with_coordinates(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_filename in image_files:
        # Form the paths for image and annotation files
        image_path = os.path.join(image_folder, image_filename)
        annotation_path = os.path.join(image_folder, os.path.splitext(image_filename)[0] + '.txt')

        # Read the image
        image = cv2.imread(image_path)
        img_height,img_width, _ = image.shape

        # Read the bounding box coordinates from the annotation file
        with open(annotation_path, 'r') as file:
            values = file.readline().strip().split()[1:]
            x, y, bbox_width, bbox_height = map(float, values)
            x *= img_width
            y *= img_height
            bbox_width *= img_width
            bbox_height *= img_height
            width, height = bbox_width*2, bbox_height+bbox_height*0.5

        

        if image is not None:
            
            x_top_left = int((x - width / 2))
            y_top_left = int((y - height / 2))

            # Calculate the bottom-right corner coordinates for the bounding box
            x_bottom_right = int((x + width / 2))
            y_bottom_right = int((y + height / 2))

            if x_top_left < 0:
                x_top_left = 0
            if y_top_left < 0:
                y_top_left = 0
            if x_bottom_right > img_width:
                x_bottom_right = int(img_width)-1
            if y_bottom_right > img_height:
                y_bottom_right = int(img_height)-1

            # if 0 <= x_top_left < img_width and 0 <= y_top_left < img_height and 0 <= x_bottom_right <= img_width and 0 <= y_bottom_right <= img_height:
            #     # Crop the image based on bounding box coordinates using float indices
            #     cropped_image = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            # else:
            #     height = img_height

            cropped_image = image[y_top_left:y_bottom_right,x_top_left:x_bottom_right]
            # Save the cropped image to the output folder
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, cropped_image)

            print(f"Cropped {image_filename} and saved to {output_path}")

# Example usage:
image_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images_cropped'