import os
import h5py
import numpy as np
import PIL
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import logging
from tqdm import tqdm
import argparse
import random   
import glob
import re
import cv2

logging.getLogger("diffusers").setLevel(logging.ERROR)

class GenAug:
    def __init__(self):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.safety_checker = lambda images, clip_input: (images, False)
    
    def aug_image(self, image, mask, obj_name, category):
        image = PIL.Image.fromarray(image).resize((512, 512))
        mask = PIL.Image.fromarray(mask).resize((512, 512))
        return self._generate_image(image, mask, obj_name)

    def _generate_image(self, image, mask, lang_prompt):
        with torch.cuda.amp.autocast(True):
            generated_image = self.pipe(prompt=lang_prompt, image=image, mask_image=mask).images[0]
        return np.array(generated_image), True

class ImageGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.genaug = GenAug()

    def generate_images(self, input_file_path, prompts, num_images=20):
        with h5py.File(input_file_path, 'r') as input_hf:
            rgb = np.array(input_hf['color'])
            masks = {
                'background': np.array(input_hf['mask_background']),
                'table': np.array(input_hf['mask_table'])
            }

        frame_rgb = rgb[0]

        for category in ['background', 'table']:
            category_dir = os.path.join(self.output_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            mask = masks[category][0].astype(np.uint8) 
            prompt = prompts[category]

            for i in tqdm(range(num_images), desc=f"Generating {category} images"):
                augmented_image = self.genaug.aug_image(frame_rgb, mask, prompt, category)
                if augmented_image.shape[:2] != mask.shape:
                    augmented_image = cv2.resize(augmented_image, (mask.shape[1], mask.shape[0]))
                
                full_image = np.zeros(frame_rgb.shape[:2] + (4,), dtype=np.uint8)
                full_image[..., 3] = mask
                full_image[mask == 255, :3] = augmented_image[mask == 1]
                
                filename = f"{prompt.replace(' ', '_')}_{i+1}.png"
                save_image(full_image, os.path.join(category_dir, filename))

        print(f"Generated images saved to {self.output_dir}")
        

class ImageCompositor:
    def __init__(self, output_dir, generated_dir):
        self.output_dir = output_dir
        self.generated_dir = generated_dir

    def composite_images(self, input_file_path, iteration):
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        
        h5_output_path = get_unique_filename(os.path.join(self.output_dir, 'h5', f'composited_{base_name}_iter{iteration}'), '.h5')
        frame_output_dir = get_unique_filename(os.path.join(self.output_dir, 'frame', f'composited_{base_name}_iter{iteration}'), '')
        
        os.makedirs(os.path.dirname(h5_output_path), exist_ok=True)
        os.makedirs(frame_output_dir, exist_ok=True)

        with h5py.File(input_file_path, 'r') as input_hf, h5py.File(h5_output_path, 'w') as output_hf:
            for key in input_hf.keys():
                input_hf.copy(key, output_hf)
            
            rgb = np.array(input_hf['color'])
            masks = {
                'background': np.array(input_hf['mask_background']),
                'table': np.array(input_hf['mask_table'])
            }

            num_frames, height, width, _ = rgb.shape
            output_hf.create_dataset('augmentation', shape=(num_frames, height, width, 3), dtype=np.uint8)

            background_image_path = select_random_image(os.path.join(self.generated_dir, 'background'))
            table_image_path = select_random_image(os.path.join(self.generated_dir, 'table'))

            background_image = process_image(np.array(Image.open(os.path.join(self.generated_dir, 'background', background_image_path))), (height, width))
            table_image = process_image(np.array(Image.open(os.path.join(self.generated_dir, 'table', table_image_path))), (height, width))
            
            
            for i in tqdm(range(num_frames), desc=f"Compositing frames for {os.path.basename(input_file_path)} (Iteration {iteration})"):
                frame_rgb = rgb[i].copy()

                background_mask = masks['background'][i] == 1
                frame_rgb[background_mask] = background_image[background_mask]
                
                table_mask = masks['table'][i] == 1
                frame_rgb[table_mask] = table_image[table_mask]

                output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
                save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))
                
            # for i in tqdm(range(num_frames), desc=f"Compositing frames for {os.path.basename(input_file_path)} (Iteration {iteration})"):
            #     frame_rgb = rgb[i].copy()

            #     background_mask = (masks['background'][i] == 1).astype(np.uint8) * 255
            #     frame_rgb = cv2.inpaint(frame_rgb, background_mask, 10, cv2.INPAINT_NS)
            #     frame_rgb[background_mask == 255] = background_image[background_mask == 255]
                
            #     table_mask = (masks['table'][i] == 1).astype(np.uint8) * 255
            #     frame_rgb = cv2.inpaint(frame_rgb, table_mask, 10, cv2.INPAINT_NS)
            #     frame_rgb[table_mask == 255] = table_image[table_mask == 255]

            #     output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
            #     save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))
            
            # for i in tqdm(range(num_frames), desc=f"Compositing frames for {os.path.basename(input_file_path)} (Iteration {iteration})"):
            #     frame_rgb = rgb[i].copy()

            #     background_mask = masks['background'][i].astype(np.float32)
            #     background_mask = cv2.GaussianBlur(background_mask, (19, 19), 0)

            #     table_mask = masks['table'][i].astype(np.float32)
            #     table_mask = cv2.GaussianBlur(table_mask, (19, 19), 0)

            #     frame_rgb = frame_rgb * (1 - background_mask[:,:,np.newaxis]) + \
            #                 background_image * background_mask[:,:,np.newaxis]

            #     frame_rgb = frame_rgb * (1 - table_mask[:,:,np.newaxis]) + \
            #                 table_image * table_mask[:,:,np.newaxis]

            #     output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
            #     save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))
            
            # for i in tqdm(range(num_frames), desc=f"Compositing frames for {os.path.basename(input_file_path)} (Iteration {iteration})"):
            #     frame_rgb = rgb[i].copy()

            #     # Process background
            #     background_mask = masks['background'][i].astype(np.float32)
            #     background_mask = cv2.GaussianBlur(background_mask, (151, 151), 0)
            #     background_mask = np.expand_dims(background_mask, axis=2)  # Add channel dimension
            #     frame_rgb = frame_rgb * (1 - background_mask) + background_image * background_mask

            #     # Process table
            #     table_mask = masks['table'][i].astype(np.float32)
            #     table_mask = cv2.GaussianBlur(table_mask, (5, 5), 0)
            #     table_mask = np.expand_dims(table_mask, axis=2)  # Add channel dimension
            #     frame_rgb = frame_rgb * (1- table_mask) + table_image * table_mask

            #     # Save the result
            #     output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
            #     save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))
            
            # for i in tqdm(range(num_frames), desc=f"Compositing frames for {os.path.basename(input_file_path)} (Iteration {iteration})"):
            #     frame_rgb = rgb[i].copy()

            #     # Process background
            #     background_mask = masks['background'][i].astype(np.float32)
            #     background_mask = cv2.GaussianBlur(background_mask, (21, 21), 0)
                
            #     # Process table
            #     table_mask = masks['table'][i].astype(np.float32)
            #     table_mask = cv2.GaussianBlur(table_mask, (11, 11), 0)
                
            #     # Combine masks
            #     combined_mask = np.maximum(background_mask, table_mask)
            #     combined_mask = np.expand_dims(combined_mask, axis=2)
                
            #     # Apply combined mask
            #     frame_rgb = frame_rgb * (1 - combined_mask) + \
            #                 (background_image * background_mask[:,:,np.newaxis] + 
            #                 table_image * table_mask[:,:,np.newaxis]) * combined_mask

            #     # Fill gaps with original image
            #     gap_mask = (1 - combined_mask) * (background_mask[:,:,np.newaxis] > 0.01) * (table_mask[:,:,np.newaxis] > 0.01)
            #     frame_rgb = frame_rgb * (1 - gap_mask) + rgb[i] * gap_mask

            #     # Save the result
            #     output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
            #     save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))

            print(f"Composited frames saved to {h5_output_path} and {frame_output_dir}")
            print(f"Used background image: {background_image_path}")
            print(f"Used table image: {table_image_path}")


def get_unique_filename(base_path, extension):
    counter = 1
    new_path = base_path
    while os.path.exists(new_path + extension):
        new_path = f"{base_path}_{counter}"
        counter += 1
    return new_path + extension

def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)

def select_random_image(directory):
    return random.choice([f for f in os.listdir(directory) if f.endswith('.png')])

def process_image(image, target_shape):
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[:2] != target_shape:
        image = np.array(Image.fromarray(image).resize(target_shape[::-1]))
    return image

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return [int(num) for num in numbers] if numbers else [0]

def process_all_files(input_dir, output_dir, generated_dir, prompts, generate=True, composite=True, iterations=1):
    if generate:
        input_file = select_input_file(input_dir)
        if input_file is None:
            print("No file selected. Exiting.")
            return
        h5_files = [input_file]
    else:
        h5_files = glob.glob(os.path.join(input_dir, "*.h5"))

    h5_files = sorted(h5_files, key=lambda x: numerical_sort(os.path.basename(x)))
    
    if not h5_files:
        print(f"No H5 files found to process")
        return

    image_generator = ImageGenerator(generated_dir)
    image_compositor = ImageCompositor(output_dir, generated_dir)

    for iteration in range(1, iterations + 1):
        if iterations > 1:
            print(f"\nStarting iteration {iteration} of {iterations}")
        for input_file in h5_files:
            print(f"\nProcessing file: {os.path.basename(input_file)}")
            
            if generate:
                image_generator.generate_images(input_file, prompts)
            
            if composite:
                image_compositor.composite_images(input_file, iteration)

def select_input_file(input_dir):
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No H5 files found in {input_dir}")
        return None

    print("Available H5 files:")
    for i, file in enumerate(h5_files):
        print(f"{i+1}. {file}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to process (or 0 to exit): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(h5_files):
                return os.path.join(input_dir, h5_files[choice-1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser(description="Generate and composite augmented images for all files in a directory.")
    parser.add_argument("--input_dir", type=str, default="/home/nrmk/Desktop/TAGen_main/data/merge", help="Directory containing input H5 files")
    parser.add_argument("--output_dir", type=str, default="augmented_output", help="Directory to save output files")
    parser.add_argument("--generated_dir", type=str, default="aug_raw", help="Directory to save generated images")
    parser.add_argument("--gen", action="store_true", help="Only generate images without compositing")
    parser.add_argument("--com", action="store_true", help="Only composite using previously generated images")
    parser.add_argument("--iterations", type=int, default=5, help="Number of times to repeat the process")
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    base_output_dir = os.path.join(os.path.dirname(current_dir), "data", args.output_dir)
    generated_dir = os.path.join(os.path.dirname(current_dir), "data", args.generated_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    prompts = {}
    if args.gen or not args.com:
        prompts = {
            'background': input("Enter the desired prompt for background: "),
            'table': input("Enter the desired prompt for table: ")
        }

    process_all_files(
        args.input_dir, 
        base_output_dir, 
        generated_dir, 
        prompts, 
        generate=args.gen or not args.com, 
        composite=not args.gen,
        iterations=1 if args.gen else args.iterations
    )

    print("All processing complete.")

if __name__ == '__main__':
    main()