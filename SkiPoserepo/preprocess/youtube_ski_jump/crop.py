import cv2
import os
import json
import numpy as np
import pickle as pkl
import pyarrow.csv as pc
import pyarrow.compute as pc_compute

csv_file = "/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/keypoints/val_new.csv"
data_path = "/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images/Validation"
output_folder_skel = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images/Validation_prova_skel'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/images_cropped/Validation'
output_folder_poses = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/youtube_skijump_dataset/skel_cropped/val.pkl'


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



video_ids = range(0,10)




# joint_coord = loaddata['gt_joints']

# joints_new = np.empty(shape=(joint_coord.shape[0],19,3))

# hips = (joint_coord[:,7,:] + joint_coord[:,10,:]) / 2
# neck = (joint_coord[:,1,:] + joint_coord[:,4,:]) / 2

# mapping = [0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]

# joints_new[:,mapping,:] = joint_coord
# joints_new[:,1,:] = neck
# joints_new[:,8,:] = hips

# Corresponding joints
joints = ['head','neck','rsho','relb','rhan','lsho','lelb','lhan','hip','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']

joints_name = ['head','rsho','relb','rhan','lsho','lelb','lhan','rhip','rkne','rank','lhip','lkne','lank','rsti','rsta','lsti','lsta']

# Bones for drawing examples
bones = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[11,15],[11,16],[9,12],[12,13],[13,14],[14,17],[14,18]]

# Check if all images exist and index them
index_list = []

len_dataset = len(os.listdir(data_path))

data = {}

poses_final = np.empty(shape=(len_dataset,len(joints),3))

# Read the CSV file into a pandas DataFrame
df = pc.read_csv(csv_file)

for i, img in enumerate(os.listdir(data_path)):
    print(img)
    poses = np.empty(shape=(len(joints_name),3))
    img_path = os.path.join(data_path,img)
    image = cv2.imread(img_path)
    h,w, _ = image.shape

    
    video_id = img.split('_')[0]
    frame_id = img.split('_')[1].split('.')[0]
    # frame_id = 70163

    condition1 = pc_compute.equal(df['event'], int(video_id))

    filtered_table = df.filter(condition1)

    columns = filtered_table.column_names

    frame_num = np.array(filtered_table['frame_num']).tolist()
    joints_columns = columns[4:56]

    condition2 = pc_compute.equal(df['frame_num'],int(frame_id))
    # Combine the conditions using logical AND
    combined_condition = pc_compute.and_(condition1, condition2)
    joints_table = df.filter(combined_condition)

    if video_id == '4' or video_id == '8' or video_id == '9':
        scale = 1.5
    else:
        scale = 1

    for column_name in joints_columns:
        index = joints_name.index(column_name.split('_')[0])
        if '_x' in column_name:
            poses[index,0] = np.array(joints_table[column_name])/scale
        elif '_y' in column_name:
            poses[index,1] = np.array(joints_table[column_name])/scale
        elif '_s' in column_name:
            poses[index,2] = np.array(joints_table[column_name])  

    joint_coord_new = np.empty(shape=(19,3))

    hips = (poses[7,:] + poses[10,:]) / 2
    neck = (poses[1,:] + poses[4,:]) / 2

    mapping = [0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]

    joint_coord_new[mapping,:] = poses
    joint_coord_new[1,:] = neck
    joint_coord_new[8,:] = hips

    skel_pix = joint_coord_new
    # skel_pix = np.empty(shape=skel.shape)
    # skel_pix[:,0] = skel[:,0]#*image.shape[1]
    # skel_pix[:,1] = skel[:,1]#*image.shape[0]
    # skel_pix[:,2] = skel[:,2]

    bounding_box = compute_bounding_box(skel_pix[:,:-1], w, h)

    # cropped_image_test = annotate_img(image, skel_pix[:,:-1],skel_pix[:,2])

    # cv2.imshow("%s" %(img),cropped_image_test)
    # cv2.waitKey(0)
    # cv2.destroyWindow("%s" %(img))
    

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

    skel_pix_bbox = np.empty(shape=skel_pix.shape)
    skel_pix_bbox[:,0] = skel_pix[:,0]-int(squared_bbox[0])
    skel_pix_bbox[:,1] = skel_pix[:,1]-int(squared_bbox[1])
    skel_pix_bbox[:,2] = skel_pix[:,2]

    poses_final[i,:,:] = skel_pix_bbox

    cropped_image_test = annotate_img(cropped_image, skel_pix_bbox[:,:-1],skel_pix_bbox[:,2])

    # cv2.imshow("cropped_image",cropped_image_test)
    # cv2.waitKey(0)

    # print('here ok')

    # poses[i,:,:] = skel_pix_bbox
    
    # # cropped_image_test = annotate_img(cropped_image, skel_pix_bbox[:,:-1],skel_pix_bbox[:,2])

    # # cv2.imshow("test cropped",cropped_image_test)
    # # cv2.waitKey(0)
    
    # # Save the cropped image to the output folder
    # output_path = os.path.join(output_folder_skel, img)
    # cv2.imwrite(output_path, cropped_image_test)

    # print(f"Cropped {image_filename} and saved to {output_path}")

    # Save the cropped image to the output folder
    # output_path = os.path.join(output_folder, img)
    # cv2.imwrite(output_path, cropped_image)

data['gt_joints'] = poses_final

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