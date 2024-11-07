import os
import cv2
from ultralytics import YOLO

def seperate_videos():
    # Define the video file path and the output folder for frames
    print("*" * 25, "separting videos to frames", "*" * 25)
    video_folder = "/datashare/project/vids_test/"

    videos = [video for video in os.listdir(video_folder) if video.endswith(".mp4")]

    for i, video_path in enumerate(videos):
        print("*" * 10, f"video number {i}", "*" * 10)
        video_name = video_path.split(".")[0]
        output_folder = f'/home/student/Desktop/Visualization_project/video_frames_project/{video_name}'

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        video = cv2.VideoCapture(os.path.join(video_folder, video_path))

        # Check if the video opened successfully
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            exit()

        frame_number = 1
        while frame_number < 1000:
            # Read the next frame from the video
            ret, frame = video.read()
            
            # If there are no more frames, break the loop
            if not ret:
                break
            
            # Construct the output path for the current frame
            frame_path = os.path.join(output_folder, f"{video_name}-frame_{frame_number}.jpg")
            
            # Save the frame as an image file
            cv2.imwrite(frame_path, frame)
            
            print(f"Saved {frame_path}")
            frame_number += 1

        # Release the video capture object
        video.release()

        print(f"All frames have been extracted to {output_folder}")

if __name__ == '__main__':
    seperate_videos()