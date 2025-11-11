import cv2
import os
import numpy as np
import skimage


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


            cropped_image = image[y_top_left:y_bottom_right,x_top_left:x_bottom_right]

            cv2.imshow("cropped_image",cropped_image)
            cv2.waitKey(0)


            # Save the cropped image to the output folder
            # output_path = os.path.join(output_folder, image_filename)
            # cv2.imwrite(output_path, cropped_image)

            print(f"Cropped {image_filename}")#and saved to {output_path}")

# Example usage:
image_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images_cropped'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

for image_filename in image_files:
    # Form the paths for image and annotation files
    # image_filename = 'export00194.png'
    image_path = os.path.join(image_folder, image_filename)
    annotation_path = os.path.join(image_folder, os.path.splitext(image_filename)[0] + '.txt')

    # Read the image
    print("Image name: ", image_filename)
    image = cv2.imread(image_path)

    # cv2.imshow("original_image",image)
    # cv2.waitKey(0)
    img_height,img_width, _ = image.shape

    # img_width,img_height, _ = image.shape

    #  [top_left(x), top_left(y), width, height].

    # Read the bounding box coordinates from the annotation file
    with open(annotation_path, 'r') as file:
        values = file.readline().strip().split()[1:]
        x, y, bbox_width, bbox_height = map(float, values)
        x *= img_width
        y *= img_height
        bbox_width *= img_width
        bbox_height *= img_height
        width, height = bbox_width*2, bbox_height+bbox_height*0.5
        center = np.array([x,y])
        width = max(bbox_width, bbox_height)
        bbox = np.array([int((x - width / 2)),int((y - height / 2)), int(width), int(height)])

        squared_bbox = make_square(bbox,center)

        x_top_left = squared_bbox[0]
        y_top_left = squared_bbox[1]

        # Calculate the bottom-right corner coordinates for the bounding box
        x_bottom_right = int((squared_bbox[0] + squared_bbox[2]))
        y_bottom_right = int((squared_bbox[1] + squared_bbox[3]))

        max_offset = 500

        image_new = np.full(shape=(img_height+max_offset,img_width+max_offset,3), fill_value = (0,0,0),dtype=np.uint8)

        image_new[max_offset:max_offset+img_height,max_offset:max_offset+img_width] = image
        cropped_image = image_new[max_offset + y_top_left:max_offset + y_bottom_right,max_offset + x_top_left:max_offset + x_bottom_right]

        # image_toplot = draw_bbox(cropped_image,squared_bbox,center)

        # cv2.imshow("cropped_image",cropped_image)
        # cv2.waitKey(0)

        # print('here ok')



        # cv2.imshow("cropped_image",cropped_image)
        # cv2.waitKey(0)


        # Save the cropped image to the output folder
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, cropped_image)
        # cv2.imshow("bbox",image_toplot)
        # cv2.waitKey(0)


        
        # padded_image = np.full((squared_bbox[2],squared_bbox[3], 3), (0,0,0), dtype=np.uint8)

        # # compute center offset
        # x_center = (squared_bbox[2] - img_width) // 2
        # y_center = (squared_bbox[3] - img_height) // 2

        # # copy img image into center of result image
        # image = padded_image[y_center:y_center+img_height, 
        #     x_center:x_center+img_width]
        
        # cropped_image = padded_image[y_top_left:y_bottom_right,x_top_left:x_bottom_right]
        
        # image_toplot = draw_bbox(padded_image,squared_bbox,center)
        # cv2.imshow("bbox",image_toplot)
        # cv2.waitKey(0)
        

        # # image_toplot = draw_bbox(padded_image,squared_bbox,center)
        # cv2.imshow("bbox",cropped_image)
        # cv2.waitKey(0)

        # (x_min, y_min, width, height)

        # scale = width / 200.0

        # input_res = 224

        # new_image = crop(image, center, scale, (input_res, input_res), rot=0)
        # cv2.imshow("new_image",new_image)
        # cv2.waitKey(0)
        # print('here')

# crop_images_with_coordinates(image_folder,output_folder)