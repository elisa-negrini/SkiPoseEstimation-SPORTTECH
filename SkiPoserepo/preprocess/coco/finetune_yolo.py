from ultralytics import YOLO
import wandb

wandb.init(project="cvsports_yolo")
model = YOLO('yolov8n-seg.pt') 
results = model.train(data='finetune_coco.yaml', epochs=100, imgsz=640, project="cvsports_yolo", name="basket_finetune")