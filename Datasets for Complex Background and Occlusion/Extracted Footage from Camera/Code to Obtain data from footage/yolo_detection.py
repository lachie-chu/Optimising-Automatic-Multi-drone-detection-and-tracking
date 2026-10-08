#Author: Lachlan Chu
#Last updated: 12/01/2026
#Purpose: Obtain the annotations of each frames with detections

from ultralytics import YOLO
import torch

print('CUDA Available:' , torch.cuda.is_available()) #Checks if the GPU is avaliable

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

model = YOLO('best.pt') #Gets the model

name = 'Videos/Drone2_8_12_2025_1150.MP4' #Name of file to test

results = model(
    name,
    conf =0.4, 
    save=True,
    save_txt=True,  # This saves each frames annotations, unless there is no annotation
    stream=False,
    show_conf=False
)