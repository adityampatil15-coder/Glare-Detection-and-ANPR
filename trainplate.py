from ultralytics import YOLO
import os
import yaml

def main():
    DATASET_PATH = r"C:\Users\adity\Desktop\DL_Projects\license_plate_project\exported_dataset"
    DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.join(DATASET_PATH, "images", "train")
    data['val']   = os.path.join(DATASET_PATH, "images", "val")
    data['nc']    = 1
    data['names'] = ['plate']

    with open(DATA_YAML, 'w') as f:
        yaml.dump(data, f)

    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        workers=4,
        name="trainv8s",
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.4,
        fliplr=0.0,
    )

if __name__ == "__main__":
    main()