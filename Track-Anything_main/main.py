"""SAVE Unit8"""

import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time
import h5py
import numpy as np

try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    
    # <Change> initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "non_clicked_masks": {},
        "fps": fps
        }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True, value=operation_log)

def run_example(example):
    return video_input

# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
        # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    if mask_dropdown:
        print("ok")
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]


    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state

def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image
    
    # Check new masking frame and 
    if "new_mask_start_frame" in interactive_state :  
        interactive_state["new_masks"].append((video_state["select_frame_number"], mask))

    operation_log = [("",""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment","Normal")]
    
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        current_mask = video_state["masks"][video_state["select_frame_number"]]
        
        # Check if the mask is already in the list
        if not any(np.array_equal(current_mask, mask) for mask in interactive_state["multi_mask"]["masks"]):
            interactive_state["multi_mask"]["masks"].append(current_mask)
            new_mask_name = f"mask_{len(interactive_state['multi_mask']['masks']):03d}"
            interactive_state["multi_mask"]["mask_names"].append(new_mask_name)
            mask_dropdown.append(new_mask_name)
        
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    
    # Return all 7 required outputs
    return (
        interactive_state,
        gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown),
        select_frame,
        [[],[]],
        operation_log,
        gr.update(visible=True),  # For mask_name_input
        gr.update(visible=True)   # For rename_mask_button
    )

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i, mask_name in enumerate(mask_dropdown):
        # Find the index of the mask in the multi_mask list
        try:
            mask_index = interactive_state["multi_mask"]["mask_names"].index(mask_name)
        except ValueError:
            # If the mask name is not found, skip it
            continue
        mask = interactive_state["multi_mask"]["masks"][mask_index]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=i+2)
    
    operation_log = [("",""), ("Select {} for tracking or inpainting".format(mask_dropdown),"Normal")]
    return select_frame, operation_log

# <Change2>
def find_first_empty_mask_frame(masks):
    for i, mask in enumerate(masks):
        if np.all(mask == 0):
            return i
    return None

# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown, save_dir):
    operation_log = [("",""), ("Track the selected masks and generate non-clicked masks for all frames.","Normal")]
    model.xmem.clear_memory()
    
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = [interactive_state["multi_mask"]["mask_names"][0]]
        mask_dropdown.sort()
        template_mask = np.zeros_like(interactive_state["multi_mask"]["masks"][0], dtype=np.uint8)
        for i, mask_name in enumerate(mask_dropdown, start=1):
            mask_index = interactive_state["multi_mask"]["mask_names"].index(mask_name)
            template_mask[interactive_state["multi_mask"]["masks"][mask_index] > 0] = i
    else:      
        template_mask = (video_state["masks"][video_state["select_frame_number"]] > 0).astype(np.uint8)
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
    
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)

    # click = (mask=1)
    updated_masks = []
    for mask in masks:
        unique_values = np.unique(mask)
        updated_mask = np.zeros_like(mask, dtype=np.uint8)
        for i, value in enumerate(unique_values[1:], start=1):  
            updated_mask[mask == value] = i
        updated_masks.append(updated_mask)
        
    masks_dict = {}
    for i, mask_name in enumerate(mask_dropdown):
        masks_dict[mask_name] = []
        for mask in updated_masks:
            masks_dict[mask_name].append((mask == i+1).astype(np.uint8))
    
    os.makedirs(save_dir, exist_ok=True)
    frame_dir = os.path.join(save_dir, "frame")
    mask_dir = os.path.join(save_dir, "mask")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    h5_filename = os.path.join(mask_dir, f"mask_{video_state['video_name']}.h5")
    save_frame(video_state, masks_dict, interactive_state, h5_filename)
    
    # consolidated_h5_filename = os.path.join(mask_dir, f"mask_{video_state['video_name']}.h5")
    # save_mask(h5_filename, consolidated_h5_filename)
    


    # Update video_state
    empty_frame_index = next((i for i, mask in enumerate(masks) if np.all(mask == 0)), None)
    
    if empty_frame_index is not None:
        last_valid_frame_index = empty_frame_index - 1
    else:
        last_valid_frame_index = len(masks) - 1
    
    frame_number = video_state["select_frame_number"] + last_valid_frame_index
    video_state["masks"][video_state["select_frame_number"]:frame_number+1] = updated_masks[:last_valid_frame_index+1]
    video_state["logits"][video_state["select_frame_number"]:frame_number+1] = logits[:last_valid_frame_index+1]
    video_state["painted_images"][video_state["select_frame_number"]:frame_number+1] = painted_images[:last_valid_frame_index+1]

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=fps)
    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(
        interactive_state["inference_times"], 
        interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
        interactive_state["positive_click_times"],
        interactive_state["negative_click_times"]
    ))
    
    empty_frame_number = int(frame_number + 1) if empty_frame_index is not None else None
    empty_visible = empty_frame_index is not None
    
    if empty_visible:
        print(f"Empty frame detected at frame {empty_frame_number}. Please add new mask.")
        operation_log.append((f"Empty frame detected at frame {empty_frame_number}. Please add new mask.", "Warning"))
    
    return video_output, video_state, interactive_state, operation_log, gr.update(value=empty_frame_number, visible=empty_visible)

def save_frame(video_state, masks_dict, interactive_state, output_path):
    with h5py.File(output_path, 'a') as hf:
        start_frame = video_state["select_frame_number"]
        last_processed_frame = start_frame

        for mask_name, masks in masks_dict.items():
            mask_dataset_name = mask_name
            stacked_masks = []

            for i, mask in enumerate(masks):
                frame_number = start_frame + i

                # If the mask is all True (255 in uint8), it's an empty frame
                if np.all(mask == 255):
                    print(f"Empty frame detected for {mask_name} at frame {frame_number}. Stopping processing for this mask.")
                    break

                stacked_masks.append(mask)
                last_processed_frame = max(last_processed_frame, frame_number)

            if stacked_masks:
                stacked_masks_array = np.stack(stacked_masks)

                if mask_dataset_name in hf:
                    del hf[mask_dataset_name]  # Delete existing dataset if it exists

                # Create dataset with user-defined name
                hf.create_dataset(mask_dataset_name, data=stacked_masks_array, dtype='uint8', compression="gzip")

                # Set attributes
                hf[mask_dataset_name].attrs['Shape'] = stacked_masks_array.shape
                hf[mask_dataset_name].attrs['Type'] = 'uint8'
                hf[mask_dataset_name].attrs['StartFrame'] = start_frame
                hf[mask_dataset_name].attrs['EndFrame'] = last_processed_frame

        # Save last processed frame number as a global attribute
        hf.attrs['last_processed_frame'] = last_processed_frame

    print(f"Masks saved to {output_path}")
    print(f"Start frame: {start_frame}, End frame: {last_processed_frame}")
    for mask_name, masks in masks_dict.items():
        print(f"Total frames for {mask_name}: {len(masks)}")
    

# def save_mask(input_path, output_path):
#     with h5py.File(input_path, 'r') as input_file, h5py.File(output_path, 'w') as output_file:
#         frame_numbers = sorted([int(key.split('_')[1]) for key in input_file.keys() if key.startswith('frame_')])
        
#         if not frame_numbers:
#             print("No frame data found in the input file.")
#             return

#         first_frame = f'frame_{frame_numbers[0]}'
#         mask_shape = input_file[first_frame].shape
        
#         masks_dataset = output_file.create_dataset('mask', shape=(len(frame_numbers), *mask_shape), dtype='uint8', compression="gzip")
        
#         for i, frame_num in enumerate(frame_numbers):
#             frame_name = f'frame_{frame_num}'
#             masks_dataset[i] = input_file[frame_name][:]

#         for attr_name, attr_value in input_file.attrs.items():
#             output_file.attrs[attr_name] = attr_value
            
#     print(f"Masks saved to {output_path}")
    
    
# <Change2>
def jump_to_empty_frame(frame_number, video_state, interactive_state):
    if frame_number is not None:
        frame_number = int(frame_number)
        if 0 <= frame_number < len(video_state["origin_images"]):
            video_state["select_frame_number"] = frame_number
            
            interactive_state["new_mask_start_frame"] = frame_number
            interactive_state["new_masks"] = []
            
            return video_state["origin_images"][frame_number], video_state, interactive_state, [("Get frame {}. Add new mask.".format(frame_number), "Info")], gr.update(visible=True, value=frame_number)
        
        
    return None, video_state, interactive_state, [("Invalid frame number.", "Error")], gr.update(visible=True,  value=1)

# inpaint 
def inpaint_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Removed the selected masks.","Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks==i] = 0
    # inpaint for videos

    try:
        inpainted_frames = model.baseinpainter.inpaint(frames, inpaint_masks, ratio=interactive_state["resize_ratio"])   # numpy array, T, H, W, 3
    except:
        operation_log = [("Error! You are trying to inpaint without masks input. Please track the selected mask first, and then press inpaint. If VRAM exceeded, please use the resize ratio to scaling down the image size.","Error"), ("","")]
        inpainted_frames = video_state["origin_images"]
    video_output = generate_video_from_frames(inpainted_frames, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def rename_mask(mask_dropdown, new_name, interactive_state):
    if not mask_dropdown or not new_name:
        return interactive_state, mask_dropdown, [("Please select a mask and enter a new name.", "Error")]
    
    old_name = mask_dropdown[0]  # Assume single selection for simplicity
    
    # Check if the new name already exists
    if new_name in interactive_state["multi_mask"]["mask_names"]:
        return interactive_state, mask_dropdown, [("This name already exists. Please choose a different name.", "Error")]
    
    idx = interactive_state["multi_mask"]["mask_names"].index(old_name)
    interactive_state["multi_mask"]["mask_names"][idx] = new_name
    
    # Update mask_dropdown
    new_dropdown = [new_name if name == old_name else name for name in mask_dropdown]
    
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=new_dropdown), [("Mask renamed successfully.", "Normal")]

if __name__ == "__main__":


    # args, defined in track_anything.py
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5800, help="The port number for the server")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the model on")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="The type of SAM model to use")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False) # not used now
    parser.add_argument("--save_dir", type=str, default="data", help="The directory to save the output")
    args = parser.parse_args()


    # check and download checkpoints if needed
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
    sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
    e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
    e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


    folder ="./checkpoints"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)


    # Save frames and masks
    current_dir = os.getcwd()
    print(current_dir)

    #save_path = os.path.join(current_dir, args.save_dir)
    save_path = os.path.join(os.path.dirname(current_dir), args.save_dir)
    print("save_path: ", save_path)

    os.makedirs(save_path, exist_ok=True)

    # initialize sam, xmem, e2fgvi models
    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)


    title = """<p><h1 align="center">Track-Anything</h1></p>
        """
    description = """<p>Gradio demo for Track Anything, a flexible and interactive tool for video object tracking, segmentation, and inpainting. I To use it, simply upload your video, or click one of the examples to load them. Code: <a href="https://github.com/gaomingqi/Track-Anything">https://github.com/gaomingqi/Track-Anything</a> <a href="https://huggingface.co/spaces/watchtowerss/Track-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""


    with gr.Blocks() as iface:
        """
            state for 
        """
        click_state = gr.State([[],[]])
        interactive_state = gr.State({
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
            "resize_ratio": 1
        }
        )

        video_state = gr.State(
            {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
            }
        )
        save_dir = gr.State(save_path)
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():

            # for user video input
            with gr.Column():
                with gr.Row(scale=0.4):
                    video_input = gr.Video(autosize=True)
                    with gr.Column():
                        video_info = gr.Textbox(label="Video Info")
                        resize_info = gr.Textbox(value="If you want to use the inpaint function, it is best to git clone the repo and use a machine with more VRAM locally. \
                                                Alternatively, you can use the resize ratio slider to scale down the original image to around 360P resolution for faster processing.", label="Tips for running this demo.")
                        resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="Resize ratio", visible=True)
                        # <Change>
                        empty_frame_number = gr.Number(label="Empty masking frame", visible=True)
                        jump_to_frame_button = gr.Button(value="Jump to Empty frame", visible=True)
                        mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                        mask_name_input = gr.Textbox(label="New mask name", placeholder="Enter new name for selected mask", visible=False)
                        rename_mask_button = gr.Button("Rename mask", visible=False)
        
            

                with gr.Row():
                    # put the template frame under the radio button
                    with gr.Column():
                        # extract frames
                        with gr.Column():
                            extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

                        # click points settins, negative or positive, mode continuous or single
                        with gr.Row():
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive",  "Negative"],
                                    value="Positive",
                                    label="Point prompt",
                                    interactive=True,
                                    visible=False)
                                remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False) 
                                clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False).style(height=160)
                                Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False)
                        template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False).style(height=360)
                        image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                        track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                
                    with gr.Column():
                        run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                        mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                        video_output = gr.Video(autosize=True, visible=False).style(height=360)
                        with gr.Row():
                            tracking_video_predict_button = gr.Button(value="Tracking", visible=False)
                            inpaint_video_predict_button = gr.Button(value="Inpainting", visible=False)

        # first step: get the video information 
        extract_frames_button.click(
            fn=get_frames_from_video,
            inputs=[
                video_input, video_state
            ],
            outputs=[video_state, video_info, template_frame,
                    image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame,
                    tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, inpaint_video_predict_button, run_status]
        )   

        # second step: select images from slider
        image_selection_slider.release(fn=select_template, 
                                    inputs=[image_selection_slider, video_state, interactive_state], 
                                    outputs=[template_frame, video_state, interactive_state, run_status], api_name="select_image")
        track_pause_number_slider.release(fn=get_end_number, 
                                    inputs=[track_pause_number_slider, video_state, interactive_state], 
                                    outputs=[template_frame, interactive_state, run_status], api_name="end_image")
        resize_ratio_slider.release(fn=get_resize_ratio, 
                                    inputs=[resize_ratio_slider, interactive_state], 
                                    outputs=[interactive_state], api_name="resize_ratio")
        
        # click select image to get mask using sam
        template_frame.select(
            fn=sam_refine,
            inputs=[video_state, point_prompt, click_state, interactive_state],
            outputs=[template_frame, video_state, interactive_state, run_status]
        )

        # add different mask
        Add_mask_button.click(
            fn=add_multi_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
        )

        remove_mask_button.click(
            fn=remove_multi_mask,
            inputs=[interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, run_status]
        )

        # tracking video from select image and mask
        tracking_video_predict_button.click(
            fn=vos_tracking_video,
            inputs=[video_state, interactive_state, mask_dropdown,save_dir],
            outputs=[video_output, video_state, interactive_state, run_status, empty_frame_number]
        )

        # inpaint video from select image and mask
        inpaint_video_predict_button.click(
            fn=inpaint_video,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[video_output, run_status]
        )

        # click to get mask
        mask_dropdown.change(
            fn=show_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[template_frame, run_status]
        )
        
        # <Change>
        jump_to_frame_button.click(
            fn=jump_to_empty_frame,
            inputs=[empty_frame_number, video_state, interactive_state],
            outputs=[template_frame, video_state, interactive_state, run_status, image_selection_slider]
        )
        # <Change>
        empty_frame_number.change(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=[empty_frame_number],
            outputs=[jump_to_frame_button]
        )
        
        rename_mask_button.click(
            fn=rename_mask,
            inputs=[mask_dropdown, mask_name_input, interactive_state],
            outputs=[interactive_state, mask_dropdown, run_status]
        )
        
        Add_mask_button.click(
            fn=add_multi_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[
                interactive_state,
                mask_dropdown,
                template_frame,
                click_state,
                run_status,
                mask_name_input,
                rename_mask_button
            ]
        )
        
        # clear input
        video_input.clear(
            lambda: (
            {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
            },
            {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": 0,
            "resize_ratio": 1
            },
            [[],[]],
            None,
            None,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
            gr.update(visible=False), gr.update(visible=False)
                            
            ),
            [],
            [ 
                video_state,
                interactive_state,
                click_state,
                video_output,
                template_frame,
                tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                Add_mask_button, template_frame, tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button,inpaint_video_predict_button, run_status
            ],
            queue=False,
            show_progress=False)

        # points clear
        clear_button_click.click(
            fn = clear_click,
            inputs = [video_state, click_state,],
            outputs = [template_frame,click_state, run_status],
        )
    iface.queue(concurrency_count=1)
    iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0", share=True)
    # iface.launch(debug=True, enable_queue=True)
