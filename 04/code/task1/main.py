from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from ptflops import get_model_complexity_info

from dataset import FacialKeypointsDataset

from models.model1 import FacialPoints

import os
import random
import importlib
import numpy as np

from torchvision.transforms import v2
from torch.utils.data import default_collate

batch_size = 64

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":  

    START_seed()  
    
    hparams = parse_arguments()             

    m = FacialPoints(hparams)

    def compute_accuracy(pred, batch): 
        if(len(batch[1].shape)==1):   
            tmp = (pred.argmax(1) == batch[1]).float()                                    
        else:
            tmp = (pred.argmax(1) == batch[1].argmax(1)).float()
        return tmp                    

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
        root_dir="./data/training_reshaped", csv_file="./data/training_frames_keypoints_resized.csv", transform=train_transform
    )
    testset = FacialKeypointsDataset(
        root_dir="./data/test_reshaped", csv_file="./data/test_frames_keypoints_resized.csv", transform=transform
    )               

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,            
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
    )
    
    #save_parameters(m, hparams)

    acc = Metric(name="accuracy", fn=compute_accuracy)    

    epochs = hparams.epochs 

    m.train(
        epochs=epochs,
        datasets={"train": train_loader, "val": test_loader},
        metrics=[acc],
        debug=hparams.debug,
    )