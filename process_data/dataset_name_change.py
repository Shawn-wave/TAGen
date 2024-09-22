import os
import h5py

def rename_datasets(file_path):
    with h5py.File(file_path, 'r+') as f:
        # 기존 데이터셋 이름과 새 이름을 매핑합니다
        name_mapping = {
            'm_b': 'mask_background',
            'm_g': 'mask_gripper',
            'm_o': 'mask_object',
            'm_t': 'mask_table'
        }
        
        for old_name, new_name in name_mapping.items():
            if old_name in f:
                f[new_name] = f[old_name]
                del f[old_name]
        
        print(f"Processed: {file_path}")

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            rename_datasets(file_path)

if __name__ == "__main__":
    folder_path = "/home/nrmk/Desktop/TAGen_main/data/merge"
    process_folder(folder_path)
    print("All files have been processed.")