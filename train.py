import os
from ultralytics import YOLO

print(os.getcwd()) #train.py 실행위치 확인

if __name__ == "__main__":
    model = YOLO('yolov8m.pt') #YOLOv8 모델을 사용
    model.train(data='waste_dataset/data.yaml', epochs=20, imgsz=640)
    model.save('model/yolov8_waste_v1.pt')