from ultralytics import YOLO

# Load the YOLO model (you may use other YOLO versions)
model = YOLO("yolo11n.pt")  # A lightweight YOLO model for faster inference

# Train the model with your dataset
model.train(data="data.yaml", epochs=5, imgsz=640, batch=16)

# The best weights will be saved as 'best.pt'. Optionally, save them as 'best.bt' as well.
model.save("best_v11.pt")  # explicitly saving as '.bt'