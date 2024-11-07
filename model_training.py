from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

results = model.train(data="/home/student/Desktop/Visualization_project/training_paths2.yml", epochs=200, device=0)