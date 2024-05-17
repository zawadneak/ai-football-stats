from ultralytics import YOLO

model = YOLO("yolov8x")

result = model.predict("videos/play3.mp4",save=True)

print(result[0])

for box in result[0].boxes:
    print(box)