from ultralytics import YOLO

model = YOLO("models/best.pt")

result = model.predict("videos/play.mp4",save=True)

print(result[0])

for box in result[0].boxes:
    print(box)