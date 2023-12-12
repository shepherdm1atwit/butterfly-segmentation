import os
import random
from ultralytics import YOLO

test_path = "./datasets/test/images/"
model = YOLO("yolov8_butterfly_custom.pt")
print("model loaded...")
for _ in range(0, 10):
    random_image = test_path + random.choice(os.listdir(test_path))
    model.predict(source=random_image, show=False, save=True)
print("Predictions complete...")
