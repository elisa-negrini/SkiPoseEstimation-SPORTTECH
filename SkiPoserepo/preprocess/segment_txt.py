from pathlib import Path

from ultralytics import SAM, YOLO


def auto_annotate_custom(data, det_model="yolov8x.pt", sam_model="sam_b.pt", device="", output_dir=None):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str, optional): Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.
        sam_model (str, optional): Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.
        device (str, optional): Device to run the models on. Defaults to an empty string (CPU or GPU, if available).
        output_dir (str | None | optional): Directory to save the annotated results.
            Defaults to a 'labels' folder in the same directory as 'data'.

    Example:
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data='ultralytics/assets', det_model='yolov8n.pt', sam_model='mobile_sam.pt')
        ```
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(data, stream=True, device=device, conf = 0.5)

    for result in det_results:
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device, conf=0.1)
            segments = sam_results[0].masks.xyn  # noqa

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:
                for i in range(len(segments)):
                    s = segments[i]
                    if len(s) == 0:
                        continue
                    segment = map(str, segments[i].reshape(-1).tolist())
                    f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")


data_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/images_resized/Validation'
output_path = '/mnt/c/Users/Giulia Martinelli/Documents/Dataset/Ski/Ski_2DPose_Dataset/segmentation_masks/Validation'
auto_annotate_custom(data=data_path, det_model="/home/giuliamartinelli/Code/SkiPose/preprocess/coco/runs/segment/train2/weights/best.pt", sam_model='sam_b.pt',output_dir=output_path)