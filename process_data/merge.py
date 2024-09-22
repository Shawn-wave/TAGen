import h5py
import os
import glob
import re

def merge_h5_files(file1_path, file2_path, output_path):
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2, h5py.File(output_path, 'w') as out:
        for name, item in f1.items():
            if isinstance(item, h5py.Dataset):
                out.create_dataset(name, data=item[:])
            elif isinstance(item, h5py.Group):
                out.create_group(name)
        
        for name, item in f2.items():
            if isinstance(item, h5py.Dataset):
                if name in out:
                    del out[name]  
                out.create_dataset(name, data=item[:])
            elif isinstance(item, h5py.Group):
                if name not in out:
                    out.create_group(name)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

def get_sorted_files(folder):
    current_dir = os.getcwd()
    base_path = os.path.join(current_dir, "..", "data", folder)
    h5_files = glob.glob(os.path.join(base_path, "*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {base_path}")
        return None

    h5_files.sort(key=lambda x: extract_number(os.path.basename(x)))
    return h5_files

def merge_all_files(mask_files, teleop_files, output_dir):
    if len(mask_files) != len(teleop_files):
        print("Error: Number of mask files and teleop files do not match.")
        return

    for i, (mask_file, teleop_file) in enumerate(zip(mask_files, teleop_files)):
        output_filename = f"merged_{i+1}.h5"
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nMerging:")
        print(f"Mask file: {os.path.basename(mask_file)}")
        print(f"Teleop file: {os.path.basename(teleop_file)}")
        print(f"Output file: {output_filename}")
        
        merge_h5_files(mask_file, teleop_file, output_path)
        print(f"Merged file created: {output_path}")

def main():
    mask_files = get_sorted_files("mask")
    teleop_files = get_sorted_files("teleop")

    if not mask_files or not teleop_files:
        print("Operation cancelled due to missing files.")
        return

    current_dir = os.getcwd()
    merge_dir = os.path.join(current_dir, "..", "data", "merge")
    os.makedirs(merge_dir, exist_ok=True)

    print("\nFiles to be merged:")
    for i, (mask_file, teleop_file) in enumerate(zip(mask_files, teleop_files)):
        print(f"\nPair {i+1}:")
        print(f"Mask: {os.path.basename(mask_file)}")
        print(f"Teleop: {os.path.basename(teleop_file)}")

    confirmation = input("\nDo you want to proceed with merging all these files? (y/n): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        return

    merge_all_files(mask_files, teleop_files, merge_dir)
    print(f"\nMerge process completed. All merged files are saved in: {merge_dir}")

if __name__ == "__main__":
    main()