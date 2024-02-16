from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  
results = model.train(data='./datasets', epochs=100, imgsz=224, batch=32)
