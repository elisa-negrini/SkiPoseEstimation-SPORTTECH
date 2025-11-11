from ultralytics import YOLO

from ultralytics.data.converter import convert_coco

convert_coco('/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/COCO_Skis/Annotations','/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/COCO_Skis/Annotations_YOLO',use_segments=True)

