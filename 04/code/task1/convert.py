
import torch

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments
from micromind import convert

from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchinfo import summary
from ptflops import get_model_complexity_info

from dataset import FacialKeypointsDataset

from models.model1_pre import FacialPoints

import os
import random
import importlib
import numpy as np

from torchvision.transforms import v2


batch_size = 200

if __name__ == "__main__":  

    hparams = parse_arguments()    


    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),                   
        ] 
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ] 
    )

    trainset = FacialKeypointsDataset(
        #root_dir="./data/training_reshaped_stretch/", csv_file="./data/training_frames_keypoints_resized.csv", transform=train_transform
        root_dir="../../data/training_reshaped_stretch/", csv_file="../../data/training_frames_keypoints_resized.csv"
    )
    testset = FacialKeypointsDataset(
        #root_dir="./data/test_reshaped_stretch/", csv_file="./data/test_frames_keypoints_resized.csv", transform=transform
        root_dir="../../data/test_reshaped_stretch/", csv_file="../../data/test_frames_keypoints_resized.csv"
    )               

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,            
    )

    batch = next(iter(train_loader))
    print(batch[0].permute(0,3,2,1).shape)
    print(type(batch[0]))

    m = FacialPoints(hparams) 


    torch.randn(10, 224, 224, 3)

    #m.export("model.tflite", "tflite", (3, 224, 224))

    m.export("model.tflite", "tflite", (3, 224, 224))

    #m.export("model.tflite", "tflite", (3, 224, 224), batch_quant=batch[0].permute(0,3,2,1))

    #convert.convert_to_tflite(m, "tflite", batch_quant=torch.randn(224,224,3))