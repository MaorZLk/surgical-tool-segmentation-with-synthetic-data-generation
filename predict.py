from ultralytics import YOLO
import os
import cv2
import shutil
import time
IMAGE_PATH = "/datashare/HW1/labeled_image_data/images/train/b2370d2c-frame_1193.jpg" # the video to be predicted on
WEIGHTS = '/home/student/Desktop/Visualization_project/best_segmentation.pt' # the weights used for predictions

'''
*******************************************************
predicts on an image, saves the predictions as output.jpg
*******************************************************
'''

def predict():
    model = YOLO(WEIGHTS)

    results = model(IMAGE_PATH, device=0, iou=0.4, augment=True)

    for result in results:
        result.save(f"output.jpg")  # save to disk
    
    print("image saved to output.jpg")


if __name__ == "__main__":
    predict()