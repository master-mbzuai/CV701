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
    
def save_parameters(model, hparams):

    path = hparams.output_folder + "/" + hparams.experiment_name

    input = (3, 32, 32)
    macs_backbone, params_backbone = get_model_complexity_info(model.modules["feature_extractor"], input, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)        
    summary_backbone = summary(model.modules["feature_extractor"], input_size=(batch_size, 3, 32, 32))        

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
        
    # train_transform = transforms.Compose(
    #     [
    #      transforms.ToTensor(), 
    #      transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)), 
    #      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    #      transforms.Resize((160, 160), antialias=True), 
    #      transforms.RandomHorizontalFlip(0.5),
    #      transforms.RandomRotation(10),         
    #     ] 
    # )
    # transform = transforms.Compose(
    #     [
    #      transforms.ToTensor(), 
    #      transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)), 
    #      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    #      transforms.Resize((160, 160), antialias=True),          
    #     ] 
    # )

    train_transform = transforms.Compose(
        [
         transforms.ToTensor(),                   
         transforms.Resize((160, 160), antialias=True),         

        ] 
    )
    transform = transforms.Compose(
        [
         transforms.ToTensor(),          
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
         transforms.Resize((160, 160), antialias=True),          
        ] 
    )
    trainset = FacialKeypointsDataset(
        root_dir="../../data/training", transform=train_transform
    )
    testset = FacialKeypointsDataset(
        root_dir="../../data/test", transform=transform
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