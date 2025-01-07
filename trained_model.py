from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import cv2
import base64
from pydantic import BaseModel

app = FastAPI()

model = YOLO('model/yolov8_trained.pt')

class ImageRequest(BaseModel):
    image: str

@app.post("/predict/")
async def predict(request: ImageRequest):
    image_data = base64.b64decode(request.image)
    image = Image.open(BytesIO(image_data))

    # YOLO 모델을 사용하여 이미지 예측
    results = model(image)
    predictions = []
    
    for result in results:
        plots = result.plot()  # 예측된 이미지 플롯
        
        class_counts = {}  # 클래스별 개수를 저장할 딕셔너리

        # 클래스 개수 세기
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            class_name = model.names[int(cls)]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        # 중복 제거된 predictions 리스트 생성
        predictions = [
            {"class": class_name, "count": count}
            for class_name, count in class_counts.items()
        ]
            
    #openCV 형식으로 변환
    _, buffer = cv2.imencode('.jpg', plots)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"predictions": predictions, "img": result_base64}


#실행 uvicorn trained_model:app --reload