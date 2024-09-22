# import h5py
# import cv2
# import numpy as np
# import os
# import glob

# def h5_to_video(h5_file_path, output_video_path, fps=30):
#     with h5py.File(h5_file_path, 'r') as hf:
#         rgb_frames = np.array(hf['color'])

#     height, width, _ = rgb_frames[0].shape

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     for frame in rgb_frames:
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(frame_bgr)

#     out.release()

#     print(f"Video saved: {output_video_path}")

# def select_input_file():
#     current_dir = os.getcwd()
#     base_path = os.path.join(os.path.dirname(current_dir), "data", "teleop")
#     h5_files = glob.glob(os.path.join(base_path, "*.h5"))
    
#     if not h5_files:
#         print(f"No H5 files found in {base_path}")
#         return None

#     print("\nAvailable H5 files:")
#     for i, file in enumerate(h5_files):
#         print(f"{i+1}. {os.path.basename(file)}")
    
#     while True:
#         try:
#             choice = int(input("Enter the number of the file you want to process (or 0 to exit): "))
#             if choice == 0:
#                 return None
#             if 1 <= choice <= len(h5_files):
#                 return h5_files[choice-1]
#             else:
#                 print("Invalid choice. Please try again.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")


        
# def main():
#     input_file = select_input_file()
#     if not input_file:
#         print("Operation cancelled.")
#         return

#     current_dir = os.getcwd()
#     output_dir = os.path.join(os.path.dirname(current_dir) ,"data", "video")
#     os.makedirs(output_dir, exist_ok=True)
    
#     output_filename = os.path.basename(input_file).replace('.h5', '.mp4')
#     output_video_path = os.path.join(output_dir, output_filename)

#     fps = 30

#     h5_to_video(input_file, output_video_path, fps)

# if __name__ == "__main__":
#     main()

#--------------------------------------------------------------------------------------------------

# import h5py
# import cv2
# import numpy as np
# import os

# def create_mp4_from_h5(h5_file_path, output_video_path, fps=30):
#     with h5py.File(h5_file_path, 'r') as h5_file:
#         if 'augmentation' not in h5_file:
#             print(f"Error: 'augmentation' dataset not found in {h5_file_path}")
#             return

#         augmentation_data = h5_file['augmentation']
#         num_frames, height, width, _ = augmentation_data.shape

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#         for i in range(num_frames):
#             frame = augmentation_data[i]
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
#             out.write(frame)

#         out.release()
    
#     print(f"Video saved to {output_video_path}")

# # 사용 예:
# h5_file_path = "/home/nrmk/Desktop/TAGen_main/data/augmented_output/h5/composited_11_6.h5"
# output_video_path = "/home/nrmk/Desktop/TAGen_main/data/augmented_output/11_6.video.mp4"
# create_mp4_from_h5(h5_file_path, output_video_path)

#------------------------------------------------------------------------------------------------------

import h5py
import cv2
import numpy as np
import os
import glob

def h5_to_video(h5_file_path, output_video_path, fps=30):
    with h5py.File(h5_file_path, 'r') as hf:
        frames = np.array(hf['mask'])

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved: {output_video_path}")

def process_all_files():
    current_dir = os.getcwd()
    base_path = "/home/nrmk/Desktop/data/delta_action/mask"
    output_dir = "/home/nrmk/Desktop/data/delta_action/mask/output"
    os.makedirs(output_dir, exist_ok=True)
    
    h5_files = glob.glob(os.path.join(base_path, "*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {base_path}")
        return

    for h5_file in h5_files:
        output_filename = os.path.basename(h5_file).replace('.h5', '.mp4')
        output_video_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {h5_file}")
        h5_to_video(h5_file, output_video_path, fps=30)

if __name__ == "__main__":
    process_all_files()