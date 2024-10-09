from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load a model
# model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
if __name__ == '__main__':
    model = YOLO("yolov8-seg.yaml", task = "segment").load('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
    model.train(
        data="ultralytics/datasets/data.yaml", 
        epochs=100, 
        imgsz=640,
        workers=2,
        batch=4
        )


# train pertama yolov8n-seg.pt