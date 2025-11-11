import cv2
import os

image_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images_cropped'
output_folder = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Skimovie_dataset/images_resized'


def resize_images(image_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_filename in image_files:
        # Form the paths for image and annotation files
        image_path = os.path.join(image_folder, image_filename)

        # Read the image
        image = cv2.imread(image_path)
        
        if image is not None:
            resized_image = cv2.resize(image, (224, 224))

            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, resized_image)

            print(f"Resized {image_filename} and saved to {output_path}")

resize_images(image_folder,output_folder)