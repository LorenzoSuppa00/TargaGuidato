from ultralytics import YOLO

# Load the exported TFLite model
tflite_model = YOLO("best_v11_float32.tflite")

# Run inference
results = tflite_model("car.jpg")