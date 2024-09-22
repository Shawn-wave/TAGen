import os
import h5py
import numpy as np
import PIL
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
import logging
from tqdm import tqdm
import argparse
import random   
import glob
import cv2
import numpy as np

logging.getLogger("diffusers").setLevel(logging.ERROR)

class GenAug:
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
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
                'background': np.array(input_hf['m_b']),
                'table': np.array(input_hf['m_t'])
            }

        frame_rgb = rgb[0]

        for category in ['background', 'table']:
            category_dir = os.path.join(self.output_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            mask = masks[category][0].astype(np.uint8) * 255
            prompt = prompts[category]

            for i in tqdm(range(num_images), desc=f"Generating {category} images"):
                augmented_image, _ = self.genaug.aug_image(frame_rgb, mask, prompt, category)
                augmented_image = np.array(Image.fromarray(augmented_image).resize(frame_rgb.shape[:2][::-1]))
                
                # RGB 이미지 생성 (검은색 배경)
                rgb_image = np.zeros(frame_rgb.shape, dtype=np.uint8)  # 검은색으로 초기화
                rgb_image[mask == 255] = augmented_image[mask == 255]  # 마스크된 영역만 채우기
                
                filename = f"{prompt.replace(' ', '_')}_{i+1}.png"
                save_image(rgb_image, os.path.join(category_dir, filename))

        print(f"Generated images saved to {self.output_dir}")

def save_image(image, path):
    Image.fromarray(image).save(path)
        
class ImageCompositor:
    def __init__(self, output_dir, generated_dir):
        self.output_dir = output_dir
        self.generated_dir = generated_dir

    def composite_images(self, input_file_path):
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        
        h5_output_path = get_unique_filename(os.path.join(self.output_dir, 'h5', f'composited_{base_name}'), '.h5')
        frame_output_dir = get_unique_filename(os.path.join(self.output_dir, 'frame', f'composited_{base_name}'), '')
        
        os.makedirs(os.path.dirname(h5_output_path), exist_ok=True)
        os.makedirs(frame_output_dir, exist_ok=True)

        with h5py.File(input_file_path, 'r') as input_hf, h5py.File(h5_output_path, 'w') as output_hf:
            for key in input_hf.keys():
                input_hf.copy(key, output_hf)
            
            rgb = np.array(input_hf['color'])
            masks = {
                'background': np.array(input_hf['m_b']),
                'table': np.array(input_hf['m_t'])
            }

            num_frames, height, width, _ = rgb.shape
            output_hf.create_dataset('augmentation', shape=(num_frames, height, width, 3), dtype=np.uint8)

            background_image_path = select_random_image(os.path.join(self.generated_dir, 'background'))
            table_image_path = select_random_image(os.path.join(self.generated_dir, 'table'))

            background_image = process_image(os.path.join(self.generated_dir, 'background', background_image_path), (height, width))
            table_image = process_image(os.path.join(self.generated_dir, 'table', table_image_path), (height, width))

            for i in tqdm(range(num_frames), desc="Compositing frames"):
                frame_rgb = rgb[i].copy()
                
                # 배경 합성
                background_mask = masks['background'][i] == 1
                frame_rgb[background_mask] = background_image[background_mask]
                
                # 테이블 합성
                table_mask = masks['table'][i] == 1
                frame_rgb[table_mask] = table_image[table_mask]

                output_hf['augmentation'][i] = frame_rgb.astype(np.uint8)
                save_image(frame_rgb, os.path.join(frame_output_dir, f'augmented_rgb_{i:04d}.png'))

            print(f"Composited frames saved to {h5_output_path} and {frame_output_dir}")
            print(f"Used background image: {background_image_path}")
            print(f"Used table image: {table_image_path}")

def process_image(image_path, target_shape):
    image = Image.open(image_path)
    image = image.convert('RGB')  # 항상 RGB 모드로 변환
    image = image.resize(target_shape[::-1])
    return np.array(image)

def select_random_image(directory):
    return random.choice([f for f in os.listdir(directory) if f.endswith('.png')])

def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)
            
def get_unique_filename(base_path, extension):
    counter = 1
    new_path = base_path
    while os.path.exists(new_path + extension):
        new_path = f"{base_path}_{counter}"
        counter += 1
    return new_path + extension

def select_input_file():
    current_dir = os.getcwd()
    base_path = os.path.join(os.path.dirname(current_dir), "data", "merge")
    h5_files = [f for f in os.listdir(base_path) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No H5 files found in {base_path}")
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
                return os.path.join(base_path, h5_files[choice-1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def process_all_files(input_dir, output_dir, generated_dir, prompts, generate=True, composite=True):
    if generate:
        input_file = select_input_file()
        if input_file is None:
            print("No file selected. Exiting.")
            return
        h5_files = [input_file]
    else:
        h5_files = glob.glob(os.path.join(input_dir, "*.h5"))
    
    if not h5_files:
        print(f"No H5 files found to process")
        return

    image_generator = ImageGenerator(generated_dir)
    image_compositor = ImageCompositor(output_dir, generated_dir)

    for input_file in h5_files:
        print(f"Processing file: {os.path.basename(input_file)}")
        
        if generate:
            image_generator.generate_images(input_file, prompts)
        
        if composite:
            image_compositor.composite_images(input_file)

def main():
    parser = argparse.ArgumentParser(description="Generate and composite augmented images for all files in a directory.")
    parser.add_argument("--input_dir", type=str, default="/home/nrmk/Desktop/TAGen_0801_test/data/merge", help="Directory containing input H5 files")
    parser.add_argument("--output_dir", type=str, default="augmented_output", help="Directory to save output files")
    parser.add_argument("--generated_dir", type=str, default="aug_raw", help="Directory to save generated images")
    parser.add_argument("--gen", action="store_true", help="Only generate images without compositing")
    parser.add_argument("--com", action="store_true", help="Only composite using previously generated images")
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    base_output_dir = os.path.join(os.path.dirname(current_dir), "data", args.output_dir)
    generated_dir = os.path.join(os.path.dirname(current_dir), "data", args.generated_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    prompts = {}
    if not args.com:
        prompts = {
            'background': input("Enter the desired prompt for background: "),
            'table': input("Enter the desired prompt for table: ")
        }

    process_all_files(
        args.input_dir, 
        base_output_dir, 
        generated_dir, 
        prompts, 
        generate=not args.com, 
        composite=not args.gen
    )

    print("Processing complete.")

if __name__ == '__main__':
    main()