from ultralytics import YOLO

# Load Backup Training model
model = YOLO("best_v11.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'
