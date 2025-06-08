import torch
from torchvision import transforms
from PIL import Image
import os
# Define the directory path of the dataset
root_dir = f"C:\\Users\\xingang\\Desktop\\outdoor"
for subdir_name in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir_name)
    subdir_path = os.path.join(subdir_path, "images")
    for foldername in os.listdir(subdir_path):
        folder_path = os.path.join(subdir_path, foldername)
        print(folder_path)
        # Loop through all files in the dataset directory
        for file_name in os.listdir(folder_path):
            # Check if the current item is a file (not a directory)
            if os.path.isfile(os.path.join(folder_path, file_name)):
                # Open the image using PIL
                img_path = os.path.join(folder_path, file_name)
                img = Image.open(img_path)
                # Convert the image to a tensor
                img_tensor = transforms.ToTensor()(img)
                # Check if the image has size torch.Size([640, 512])
                if img_tensor.shape == torch.Size([1, 640, 512]):
                    print(f"abnormal img shape found for file: {img_path} with shape: {img_tensor.shape}")
                    # Rotate the image by 90 degrees clockwise
                    rotated_img_tensor = torch.rot90(img_tensor, 1, [1, 2])
                    # Save the rotated image
                    rotated_img = transforms.ToPILImage()(rotated_img_tensor)
                    rotated_img.save(img_path)
