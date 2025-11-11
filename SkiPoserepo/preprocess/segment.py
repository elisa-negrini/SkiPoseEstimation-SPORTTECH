import cv2
from ultralytics import YOLO
import numpy as np
import torch


# img= cv2.imread('/home/giuliamartinelli/Code/SkiPose/preprocess/bus.jpg')
# model = YOLO('yolov8m-seg.pt')
# results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
# for result in results:
#     # get array results
#     masks = result.masks.data
#     boxes = result.boxes.data
#     # extract classes
#     clss = boxes[:, 5]
#     # get indices of results where class is 0 (people in COCO)
#     people_indices = torch.where(clss == 0)
#     # use these indices to extract the relevant masks
#     people_masks = masks[people_indices]
#     # scale for visualizing results
#     people_mask = torch.any(people_masks, dim=0).int() * 255
#     # save to file
#     # cv2.imshow('mask', people_mask.cpu().numpy().astype(np.uint8))
#     # cv2.waitKey(0)
#     cv2.imwrite('/home/giuliamartinelli/Code/SkiPose/preprocess/merged_segs.jpg', people_mask.cpu().numpy())


# Load the original image
image = cv2.imread('/home/giuliamartinelli/Code/SkiPose/preprocess/bus.jpg')

# Load the mask image (grayscale)
mask = cv2.imread('/home/giuliamartinelli/Code/SkiPose/preprocess/merged_segs.jpg', cv2.IMREAD_GRAYSCALE)

scaleh, scalew = image.shape[0]/mask.shape[0], image.shape[1]/mask.shape[1]

mask = cv2.resize(image, (mask.shape[1] * scalew, mask.shape[0] * scaleh))

mask = mask.astype('uint8')

# Convert mask to a binary mask (optional, depending on your mask)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Invert the binary mask (optional, depending on your mask)
inverted_mask = cv2.bitwise_not(binary_mask)

# Create a white background image
background = np.ones_like(image) * 255

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=inverted_mask)
background_masked = cv2.bitwise_and(background, background, mask=binary_mask)

# Combine the masked image with the background
result = cv2.add(masked_image, background_masked)

# Display the result
cv2.imshow('Masked Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()