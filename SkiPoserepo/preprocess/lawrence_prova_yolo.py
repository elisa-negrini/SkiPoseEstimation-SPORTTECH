from ultralytics import YOLO
import torch
import cv2

model = YOLO('/home/giuliamartinelli/Code/SkiPose/preprocess/coco/runs/segment/train2/weights/best.pt')
img_path = "/home/giuliamartinelli/Code/SkiPose/preprocess/ski_jump2.jpg"

results = model(img_path,save=False)

for result in results:
    
    boxes = result.boxes.data

    ###BBOX COORDINATES FORMAT: [x_center, y_center, width, height, confidence, class]

    clss = boxes[:, 5]
    people_indices = torch.where(clss == 0)
    skis_indices = torch.where(clss==1)

    print(people_indices)
    
    if skis_indices == None:
        print("No skis Detected")
    else:
        ski_boxes = boxes[skis_indices]
        areas_skis = []
        print("ski boxes: ", ski_boxes)
        for ski_box in ski_boxes:
            width_ski = ski_box[2]      # Get the width
            height_ski = ski_box[3]     # Get the height
            area_ski = width_ski * height_ski   # Calculate the area
            areas_skis.append(area_ski)     # save the area in a list

        sorted_areas = sorted(areas_skis, reverse=True) # order the areas from the heighest to the lowest

        ski_boxes = sorted_areas[:2]    # the the two largest areas
    
        ski_box1_index = areas_skis.index(ski_boxes[0]) # Get the index of the first largest area
        ski_box2_index = areas_skis.index(ski_boxes[1]) # Gest the index of the second largest area

        print("index 1: ",ski_box1_index)
        print("index 2: ",ski_box2_index)

        largest_boxes = [ski_boxes[ski_box1_index],ski_boxes[ski_box2_index]]   # Get the two ski boxes
        print("ski largest boxes: ",largest_boxes)
        

        ##Calculate the area of the bbox (width x height). The two heighest areas are the skis bboxes.


    if people_indices == None:
        print("No person Detected")
    else:
        person_boxes = boxes[people_indices]
        print("person box: ", person_boxes)

        ##Calculate the distance between all person bbox center and the image center. The closest one is the skier.