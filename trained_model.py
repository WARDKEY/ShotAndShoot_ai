from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import base64
import os

app = FastAPI()

print(os.getcwd())
model = YOLO('model/yolov8_trained.pt')

@app.post("/predict/")
async def predict(file: UploadFile):
    image_bytes = await file.read()

    # 바이트 데이터를 PIL 이미지로 변환
    image = Image.open(BytesIO(image_bytes))

    # YOLO 모델을 사용하여 이미지 예측
    results = model(image)
    #results = model("C:/Users/dndsm/Desktop/plastic_bottle/plastic2.jpg")

    predictions = results[0].names[0]  #클래스 이름
    plots = results[0].plot() #예측 사진

    _, buffer = cv2.imencode('.jpg', plots)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"class": predictions, "image": result_base64}

#실행 uvicorn trained_model:app --reload