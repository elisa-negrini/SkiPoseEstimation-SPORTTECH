from ultralytics import YOLO
import cv2
import torch
import torchvision
from torchvision.transforms.functional import invert
from torchvision import transforms
import numpy as np
import os

data_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images_cropped/Validation'
output_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images_segm/Validation'
output_imagesegm_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images_masked/Validation'

model = YOLO('/home/giuliamartinelli/Code/SkiPose/preprocess/coco/runs/segment/train6/weights/best.pt')

for img in os.listdir(data_path):
    img_path = os.path.join(data_path,img)
    results = model(img_path,save=True)
    original_image = cv2.imread(img_path)
    original_height, original_width, _ = original_image.shape

    for result in results:
        if result.masks is None:
            cv2.imwrite(os.path.join(output_path, img), original_image*0)
            cv2.imwrite(os.path.join(output_imagesegm_path, img), original_image*0)
        else:
            masks = result.masks.data
            image = torch.tensor(result.orig_img).cuda()
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            skis_indices = torch.where(clss==1)
            if skis_indices == None:
                ski_mask = torch.any(masks, dim=0).int() * 0
            else:
                ski_masks = masks[skis_indices]
                ski_mask = torch.any(ski_masks, dim=0).int() * 122
                rgb_segm_skimask = torch.any(ski_masks, dim=0).int() * 255

            if people_indices == None:
                people_mask = torch.any(masks, dim=0).int() * 0
            else:
                people_masks = masks[people_indices]
                people_mask = torch.any(people_masks, dim=0).int() * 255

            ski_mask = ski_mask + people_mask

            rgb_segm_mask = rgb_segm_skimask + people_mask

            # Resize masks to the original image resolution
            ski_mask_resized = cv2.resize(ski_mask.cpu().numpy().astype(np.uint8), (original_width, original_height))
            rgb_segm_mask_resized = cv2.cvtColor(cv2.resize(rgb_segm_mask.cpu().numpy().astype(np.uint8), (original_width, original_height)), cv2.COLOR_GRAY2BGR)

            image_masked = cv2.bitwise_and(original_image, rgb_segm_mask_resized)

            # Save the resized masks
            cv2.imwrite(os.path.join(output_path, img), ski_mask_resized)
            cv2.imwrite(os.path.join(output_imagesegm_path, img), image_masked)
            

            # segmented_image = image + rgb_segm_mask
            # save to file
            #cv2.imwrite(os.path.join(output_path,img), ski_mask_resized.cpu().numpy())
            # cv2.imwrite(os.path.join(output_imagesegm_path,img), masks.cpu().numpy())
