import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import natsort

# get this parameters from tuning file used for exposure sweep
exp_index_step = 3.0
max_exposure = 15 * 11000
min_exposure = 1 * 20
max_exp_index = 20 * np.log2(max_exposure / min_exposure)
index = 5
scenario = "bright"

file_list_cam0 = glob.glob(f"C:/Users/xingang/Desktop/AEC_dataset/ExpoSweep/evaluation/indoor/{scenario}_scene{index}/*.png")
file_list_cam0 = natsort.natsorted(file_list_cam0)
file_list_cam1 = glob.glob(f"C:/Users/xingang/Desktop/AEC_dataset/ExpoSweep/evaluation/indoor/{scenario}_scene{index}/*.png")
file_list_cam1 = natsort.natsorted(file_list_cam1)

# create output path
output_path = f"./raw/raw_indoor_{scenario}_{index}"
os.makedirs(output_path, exist_ok=False)

for idx, file in enumerate(file_list_cam0):
    img = imageio.imread(file)

    h, w = img.shape[0:2]
    s = w
    expIndex = exp_index_step * idx
    expIndex = min(expIndex, max_exp_index)
    outputFile = r'frame_w{}_h{}_s{}_index{:03d}_cam0_expIndex_{:.1F}.raw'.format(w, h, s, idx, expIndex)
    img.tofile(os.path.join(output_path, outputFile))
    
for idx, file in enumerate(file_list_cam1):
    img = imageio.imread(file)
    h, w = img.shape[0:2]
    s = w
    expIndex = exp_index_step * idx
    expIndex = min(expIndex, max_exp_index)
    outputFile = r'frame_w{}_h{}_s{}_index{:03d}_cam1_expIndex_{:.1F}.raw'.format(w, h, s, idx, expIndex)
    img.tofile(os.path.join(output_path, outputFile))
