#importing libraries

import os
import numpy as np
import torch
import time
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import cv2
import glob
from PIL import Image
import ntpath
import os
from tqdm import tqdm
from tqdm import trange

from dataset import ImageWoof
from model import CNN
from parser import parse_arguments
from matplotlib import pyplot as plt

from torchinfo import summary
from pathlib import Path

params = parse_arguments()

lr=params.lr
epochs=params.epochs
optimizer=params.opt
batch_size = 64
path=params.output_folder + "_e" + str(params.epochs) + "_l" + str(params.lr) + "_" + str(params.opt) + "_" + str(params.batch_size)
device = 'cuda'

Path(path).mkdir(exist_ok=True)

with open(path + "/" + "meta.txt", "w+") as file:
    file.write(str(params))

def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((160, 160), antialias=True)
        ]
    )

trainset = ImageWoof(
    root="../", train=True, transform=transform, img_size=160
)
testset = ImageWoof(
    root="../", train=False, transform=transform, img_size=160
)

## split into train, val, test 
print(len(trainset))     
val_size = int(0.1 * len(trainset))
print(val_size)
train_size = len(trainset) - val_size
train, val = torch.utils.data.random_split(trainset, [train_size, val_size])    

train_loader = torch.utils.data.DataLoader(
    train, batch_size=batch_size, shuffle=True, num_workers=1
)
val_loader = torch.utils.data.DataLoader(
    val, batch_size=batch_size, shuffle=False, num_workers=1
)    
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=1
)

# train_loader.to(device)
# val_loader.to(device)
# test_loader.to(device)

print("Trainset size: ", len(train)//batch_size)
print("Valset size: ", len(val)//batch_size)
print("Testset size: ", len(testset)//batch_size)

## Creating training loop
def train(model, epoch):
    model.train()
    train_loss = 0
    total = 0
    correct=0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            # send to device
            data, target = data.to(device), target.to(device)
    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            tepoch.set_postfix(loss=train_loss/(batch_idx+1), lr=optimizer.param_groups[0]['lr'])
        log = '{} train loss: {:.4f} accuracy: {:.4f}\n'.format(epoch, train_loss/(batch_idx+1), 100.*correct/total)
        print(log)
        with open(path + "/log.txt", 'a') as file:
                file.write(log)

best_accuracy = 0.0
def validate(model):
    global best_accuracy
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            data, target = data.to(device), target.to(device)
    
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        if (100.*correct/total) > best_accuracy:
            print("Saving the best model...")
            best_accuracy = (100.*correct/total)
            torch.save(model.state_dict(), path + '/best_model_adam.pth')
            log = ' val loss: {:.4f} accuracy: {:.4f} best_accuracy: {:.4f}\n'.format(test_loss/(batch_idx+1), 100.*correct/total, best_accuracy)
            with open(path + "/log.txt", 'a') as file:
                file.write(log)

        print(log)


if __name__ == "__main__":

    START_seed()
    #train_loader, val_loader, test_loader = load_dataset()

    model = CNN(num_classes=10)
    pytorch_total_params = sum(p.numel() for p in  model.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))

    criterion = nn.CrossEntropyLoss()
    if(params.opt == "adam"):            
        optimizer = optim.Adam(model.parameters(), lr=params.lr)
    elif(params.opt == "adamW"):
        optimizer = optim.AdamW(model.parameters(), lr=params.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.lr)

    model.to(device)
    start = time.time()

    for epoch in range(0, epochs):
        print("epoch number: {0}".format(epoch))
        train(model, epoch)
        validate(model)
    end = time.time()
    Total_time=end-start
    print('Total training and inference time is: {0}'.format(Total_time))