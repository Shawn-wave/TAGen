import h5py
import numpy as np

def analyze_h5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print("H5 File Structure:")
        
        frame_count = 0
        mask_shape = None
        
        for name, item in f.items():
            if isinstance(item, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {item.shape}")
                print(f"  Type: {item.dtype}")
                
                if name.startswith('frame_'):
                    frame_count += 1
                elif name == 'mask':
                    mask_shape = item.shape
        
        # print("\nFile Attributes:")
        # for attr_name, attr_value in f.attrs.items():
        #     print(f"  {attr_name}: {attr_value}")
        
        # print(f"\nTotal number of individual frames: {frame_count}")
        
        # if mask_shape:
        #     print(f"Mask dataset shape: {mask_shape}")
        #     print(f"Number of frames in mask dataset: {mask_shape[0]}")
        
        # total_frames = max(frame_count, mask_shape[0] if mask_shape else 0)
        # print(f"\nTotal number of frames (including mask dataset): {total_frames}")

if __name__ == "__main__":
    h5_file_path = '/home/nrmk/Desktop/data/delta_action/mask/mask_29.mp4.h5'  
    # h5_file_path = '/home/nrmk/Desktop/TAGen_0801/data/merge/11.h5'  
    analyze_h5_structure(h5_file_path)

# import h5py
# import numpy as np

# import h5py
# import numpy as np

# def explore_h5_datasets(file_path):
#     with h5py.File(file_path, 'r') as f:
#         for name, obj in f.items():
#             if isinstance(obj, h5py.Dataset):
#                 print(f"\nDataset: {name}")
#                 print(f"  Shape: {obj.shape}")
#                 print(f"  Type: {obj.dtype}")
#                 if len(obj.attrs) > 0:
#                     print("  Attributes:")
#                     for attr_name, attr_value in obj.attrs.items():
#                         if isinstance(attr_value, np.ndarray):
#                             if len(attr_value) > 10:
#                                 print(f"    {attr_name}: [{attr_value[0]}, {attr_value[1]}, ..., {attr_value[-1]}]")
#                             else:
#                                 print(f"    {attr_name}: {list(attr_value)}")
#                         else:
#                             print(f"    {attr_name}: {attr_value}")
#                 else:
#                     print("  No attributes found for this dataset.")

# # 사용 예시
# # explore_h5_datasets("/home/nrmk/Desktop/GenAug/mask_data/masks_0.mp4.h5")
# explore_h5_datasets("/home/nrmk/Desktop/TAM/Track-Anything_0725_test/logs/masks_16.mp4.h5")

