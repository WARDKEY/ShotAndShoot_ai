import os
from ultralytics import YOLO

print(os.getcwd()) #train.py 실행위치 확인

if __name__ == "__main__":
    model = YOLO('yolov8m.pt')
    model.train(data='waste_data/data.yaml', epochs=2, imgsz=640)
    model.save('model/yolov8_trained.pt')


