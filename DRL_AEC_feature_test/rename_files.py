import os
from PIL import Image
import numpy as np

def rename_subfolders(parent_folder):
    # List all subfolders in the parent directory
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    
    
    # # Sort subfolders to ensure consistent ordering
    subfolders.sort()

    # dim_num = 0
    # netural_num = 0
    # bright_num = 0
    
    # Rename each subfolder
    for index, subfolder in enumerate(subfolders, start=1):

        # mid_image_path = os.path.join(parent_folder, f'{subfolder}\\50.png')
        # # Read the image
        # middle_image = Image.open(mid_image_path)

        # # Convert PIL Image to NumPy array
        # middle_image = np.array(middle_image)

        # # Calculate average luma
        # average_luma = np.mean(middle_image)



        # if 0 <= average_luma <= 80:
        #     new_name = f"dim_scene{index}"
        #     dim_num += 1
        # elif 80 < average_luma <= 120:
        #     new_name = f"neutral_scene{index}"
        #     netural_num += 1
        # elif 120 < average_luma:
        #     new_name = f"bright_scene{index}"
        #     bright_num += 1

        new_name = f"neutral_scene{index}"
        old_path = os.path.join(parent_folder, subfolder)
        new_path = os.path.join(parent_folder, new_name)
        
        # Rename the folder
        os.rename(old_path, new_path)
        print(f"Renamed '{subfolder}' to '{new_name}'")

    # print(dim_num)
    # print(netural_num)
    # print(bright_num)


    # folder_path = f"C:\\Users\\xingang\\Desktop\\outdoor"
    # # List all files in the directory
    # files = os.listdir(folder_path)
    # # Loop through each file in the directory
    # for file_name in files:
    #     # Split the file name to extract the number
    #     parts = file_name.split('-')
    #     # print(parts)
    #     if len(parts) > 2:
    #         number = parts[2]
    #         # Remove leading zeros
    #         new_number = str(int(number))
    #         # Create the new file name
    #         new_file_name = f"{new_number}.png"
    #         # Construct full file paths
    #         old_file_path = os.path.join(folder_path, file_name)
    #         new_file_path = os.path.join(folder_path, new_file_name)
    #         # Rename the file
    #         os.rename(old_file_path, new_file_path)
    
    # files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
root_dir = f"C:\\Users\\xingang\\Desktop\\AEC_dataset\\ExpoSweep\\training\\neutral"
rename_subfolders(root_dir)
# for subdir_name in os.listdir(root_dir):
#     subdir_path = os.path.join(root_dir, subdir_name)
#     subdir_path = os.path.join(subdir_path, "images")
#     print(subdir_path)
#     for foldername in os.listdir(subdir_path):
#         folder_path = os.path.join(subdir_path, foldername)
#         print(folder_path)
#         for file_name in os.listdir(subdir_path):
#             # List the remaining files after deletion
#             remaining_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
#             # Rename the remaining files
#             for index, file_name in enumerate(remaining_files, start=1):
#                 # Create the new file name
#                 new_file_name = f"{index}.png"
#                 # Construct full file paths
#                 old_file_path = os.path.join(folder_path, file_name)
#                 new_file_path = os.path.join(folder_path, new_file_name)
#                 # Rename the file
#                 os.rename(old_file_path, new_file_path)


#             print("Files have been renamed successfully.")
