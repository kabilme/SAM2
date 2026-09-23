# YOLOv8 Instance Segmentation Dataset

- Total Images: 4
- Total Objects: 4
- Splits: {'train': 3, 'val': 1}
- Classes: {'scooter': 4, 'person': 0, 'helmet': 0}

## Training with Ultralytics YOLO:
```bash
yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=50
```
