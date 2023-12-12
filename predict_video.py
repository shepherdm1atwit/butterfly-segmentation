from ultralytics import YOLO

model = YOLO("yolov8_butterfly_custom.pt")
print("model loaded...")

model.predict(source="./butterfly.mp4", show=True, save=True)
