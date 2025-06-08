import os
import shutil


root_dir = "C:\\Users\\xingang\\Desktop\\outdoor"

for subdir_name in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir_name)
    subdir_path = os.path.join(subdir_path, "images")
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.startswith('205-') and filename.endswith(".png"):
                # Extract the values of i and j from the filename
                parts = filename.split('-')
                i = parts[1]
                j = parts[2].split('.')[0]
                # Create the new directory
                new_dir = os.path.join(subdir_path, f'angle_{i}')
                os.makedirs(new_dir, exist_ok=True)
                # Move the file to the new directory and rename it
                new_file_name = f'{int(j)}.png'
                shutil.move(os.path.join(subdir_path, filename), os.path.join(new_dir, new_file_name))
