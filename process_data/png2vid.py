import cv2
import os
import numpy as np
from glob import glob

def create_video_from_images(image_folder, output_video_name, fps=30):
    # Get list of PNG files in the folder
    images = sorted(glob(os.path.join(image_folder, '*.png')))
    
    if not images:
        print(f"No PNG images found in {image_folder}")
        return
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    # Release the VideoWriter
    video.release()

    print(f"Video saved as {output_video_name}")

if __name__ == "__main__":
    # Folder containing PNG images
    image_folder = '/home/nrmk/Desktop/TAGen_main/data/augmented_output/frame/composited_11_5'
    
    # Output video file name
    output_video = '/home/nrmk/Desktop/TAGen_main/data/augmented_output/frame/11_5.video.mp4'
    
    # Frames per second for the output video
    fps = 30

    create_video_from_images(image_folder, output_video, fps)
    
    