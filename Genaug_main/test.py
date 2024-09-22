import os
import h5py
import numpy as np
import PIL
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import logging
from tqdm import tqdm
import glob
import argparse

# click = 1
logging.getLogger("diffusers").setLevel(logging.ERROR)

class GenAug:
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.safety_checker = lambda images, clip_input: (images, False)
        
    # def __init__(self):
    #     self.pipe = StableDiffusionPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-v1-5",
    #         torch_dtype=torch.float16,
    #     ).to("cuda")

    def aug_background(self, image, mask, obj_name):
        # image = rgb[0]
        image = PIL.Image.fromarray(image).resize((512, 512))
        mask = PIL.Image.fromarray(mask).resize((512, 512))
        image, _ = self.get_background(image, mask, obj_name)
        return image

    def aug_table(self, image, mask, obj_name):
        # image = rgb[0]
        image = PIL.Image.fromarray(image).resize((512, 512))
        mask = PIL.Image.fromarray(mask).resize((512, 512))
        image, _ = self.get_table(image, mask, obj_name)
        return image

    def get_background(self, image, mask, new_name):
        image = np.array(image)
        zoomed_mask = PIL.Image.fromarray(np.array(mask))
        zoomed_rgb = PIL.Image.fromarray(image)
        lang_prompt = new_name

        with torch.cuda.amp.autocast(True):
            generated_image = self.pipe(prompt=lang_prompt, image=zoomed_rgb, mask_image=zoomed_mask).images[0]

        return np.array(generated_image), True

    def get_table(self, image, mask, new_name):
        image = np.array(image)
        zoomed_mask = PIL.Image.fromarray(np.array(mask))
        zoomed_rgb = PIL.Image.fromarray(image)
        lang_prompt = new_name

        with torch.cuda.amp.autocast(True):
            generated_image = self.pipe(prompt=lang_prompt, image=zoomed_rgb, mask_image=zoomed_mask).images[0]

        return np.array(generated_image), True
    
def process_h5_file(input_file_path, output_file_path, output_dir, prompts):
    with h5py.File(input_file_path, 'r') as input_hf, h5py.File(output_file_path, 'w') as output_hf:
        # Copy all datasets from input to output
        for key in input_hf.keys():
            input_hf.copy(key, output_hf)
            
        rgb = np.array(input_hf['color'])
        masks = {
            'background': np.array(input_hf['m_b']),
            'table': np.array(input_hf['m_t'])
        }

        print(f"RGB shape: {rgb.shape}")
        for mask_name, mask in masks.items():
            print(f"{mask_name.capitalize()} mask shape: {mask.shape}")
            
        genaug = GenAug()

        num_frames = rgb.shape[0]

        # Create a new dataset for augmented images
        output_hf.create_dataset('augmentation', shape=(num_frames, rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)


        masks['background'] = masks['background'].astype(np.uint8) * 255
        masks['table'] = masks['table'].astype(np.uint8) * 255

        # Generate full background and table images using the first frame
        frame_rgb = rgb[0]
        background_mask = masks['background'][0]
        table_mask = masks['table'][0]

        # Generate full background
        full_background = genaug.aug_background(frame_rgb, background_mask, prompts['background'])
        full_background = np.array(Image.fromarray(full_background).resize((frame_rgb.shape[1], frame_rgb.shape[0])))

        # Generate full table
        full_table = genaug.aug_table(frame_rgb, table_mask, prompts['table'])
        full_table = np.array(Image.fromarray(full_table).resize((frame_rgb.shape[1], frame_rgb.shape[0])))
        
        input_file_name = os.path.basename(input_file_path)

        for i in tqdm(range(num_frames)):
            frame_rgb = rgb[i]
            augmented_frame = frame_rgb.copy()

            for mask_name, mask in masks.items():
                frame_mask = mask[i]
                if mask_name == 'background':
                    mask_bool = (frame_mask == 255)
                    if np.any(mask_bool):
                        augmented_frame[mask_bool] = full_background[mask_bool]
                
                elif mask_name == 'table':
                    mask_bool = (frame_mask == 255)
                    if np.any(mask_bool):
                        augmented_frame[mask_bool] = full_table[mask_bool]

            # Save the augmented frame to the new H5 file
            output_hf['augmentation'][i] = augmented_frame

            # Save and display the result for this frame
            save_and_display_frame(augmented_frame, i, output_dir, input_file_name)

            # Yield the progress
            yield i + 1, num_frames

        print(f"Augmented data saved to {output_file_path} and individual frames saved to {output_dir}")


def save_and_display_frame(rgb, frame_num, aug_dir, input_file_name):
    # Create a subdirectory with the input file name (without extension)
    file_name_without_ext = os.path.splitext(input_file_name)[0]
    save_dir = os.path.join(aug_dir, file_name_without_ext)
    os.makedirs(save_dir, exist_ok=True)
    
    rgb_path = os.path.join(save_dir, f'augmented_rgb_{frame_num:04d}.png')
    Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)
    
def select_input_file():
    current_dir = os.getcwd()
    base_path = os.path.join(os.path.dirname(current_dir), "data", "merge")
    h5_files = glob.glob(os.path.join(base_path, "*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {base_path}")
        return None

    print("Available H5 files:")
    for i, file in enumerate(h5_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to process (or 0 to exit): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(h5_files):
                return h5_files[choice-1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate augmented images with custom backgrounds.")
    parser.add_argument("--save_dir", type=str, default="aug_rgb", help="Directory to save output files")
    parser.add_argument("--input_file", type=str, help="Input H5 file path")
    args = parser.parse_args()
    
    input_h5_file = select_input_file()
    if input_h5_file is None:
        print("No file selected. Exiting.")
        exit()

    print(f"Selected file: {input_h5_file}")

    current_dir = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_dir), "data", args.save_dir)
    os.makedirs(save_path, exist_ok=True)
    print("Save path for individual frames:", save_path)

    output_dir = os.path.join(os.path.dirname(current_dir), "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(input_h5_file))  # 이 줄을 수정
    print(f"Output H5 file path: {output_file_path}")
    
    prompts = {
        'object': 'object prompt',
        'background': 'background prompt',
        'table': 'table prompt'
    }
    
    for mask_name in ['background', 'table']:
        prompts[mask_name] = input(f"Enter the desired prompt for {mask_name}: ")
        
    for progress in process_h5_file(input_h5_file, output_file_path, save_path, prompts):
        print(f"Processed {progress[0]} out of {progress[1]} frames")
    
