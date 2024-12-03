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
    predictions = []
    
    for result in results:
        plots = result.plot() # 예측된 이미지 플롯
        
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            predictions.append({
                "class": model.names[int(cls)], 
                "reliability": round(float(conf), 2),
            })
            
    #openCV 형식으로 변환
    _, buffer = cv2.imencode('.jpg', plots)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"predictions": predictions, "img": result_base64}


#실행 uvicorn trained_model:app --reload