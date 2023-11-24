from micromind import Metric
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchinfo import summary
from ptflops import get_model_complexity_info

from dataset import FacialKeypointsDataset

from models.model2 import FacialPoints

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

def save_parameters(model, hparams):

    path = hparams.output_folder + "/" + hparams.experiment_name

    input = (3, 32, 32)
    macs_backbone, params_backbone = get_model_complexity_info(model.modules["feature_extractor"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_backbone = summary(model.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))    
    #print(summary_backbone)

    input = (model.input, 1, 1)
    macs_classifier, params_classifier = get_model_complexity_info(model.modules["classifier"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_classifier = summary(model.modules["classifier"], input_size=(10, model.input, 1, 1))    

    output = "BACKBONE\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_backbone, params_backbone)
    output += str(summary_backbone) + "\n"
    output += "\n"*2
    output += "CLASSIFIER\n" 
    output += "MACs {}, learnable parameters {}\n".format(macs_classifier, params_classifier)
    output += str(summary_classifier)        

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/architecture.txt', 'w') as file:
        file.write(output)    

    with open(path + '/meta.txt', 'w') as file:
        file.write(str(hparams))

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
        root_dir="./data/training_reshaped_stretch/", csv_file="./data/training_frames_keypoints_resized.csv", transform=train_transform
    )
    testset = FacialKeypointsDataset(
        root_dir="./data/test_reshaped_stretch/", csv_file="./data/test_frames_keypoints_resized.csv", transform=transform
    )               

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,            
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, 
        shuffle=False, 
        num_workers=1,
    )
    
    save_parameters(m, hparams)

    acc = Metric(name="accuracy", fn=compute_accuracy)    

    epochs = hparams.epochs 

    m.train(
        epochs=epochs,
        datasets={"train": train_loader, "val": test_loader},
        metrics=[acc],
        debug=hparams.debug,
    )