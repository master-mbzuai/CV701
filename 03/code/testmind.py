from micromind import MicroMind, Metric
from micromind.networks import PhiNet
from micromind.utils.parse import parse_arguments

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from dataset import ImageWoof
from matplotlib import pyplot as plt

from torchinfo import summary

batch_size = 128

class ImageClassification(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules["classifier"] = PhiNet(
            (3, 160, 160), include_top=True, num_classes=10, alpha=2.1
        )

    def forward(self, batch):
        return self.modules["classifier"](batch[0])

    def compute_loss(self, pred, batch):
        return nn.CrossEntropyLoss()(pred, batch[1])


if __name__ == "__main__":
    hparams = parse_arguments()
    hparams.output_folder = 'test_3'
    m = ImageClassification(hparams)

    summary(m.modules["classifier"], input_size=(batch_size, 3, 160, 160))


    def compute_accuracy(pred, batch):
        tmp = (pred.argmax(1) == batch[1]).float()
        return tmp

    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize((160, 160), antialias=True)
        ]
    )

    trainset = ImageWoof(
        root=".", train=True, transform=transform, img_size=160
    )
    testset = ImageWoof(
        root=".", train=False, transform=transform, img_size=160
    )

    ## split into train, val, test 
    print(len(trainset))     
    val_size = int(0.1 * len(trainset))
    print(val_size)
    train_size = len(trainset) - val_size
    train, val = torch.utils.data.random_split(trainset, [train_size, val_size])    

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=1
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=1
    )    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    print("Trainset size: ", len(train)//batch_size)
    print("Valset size: ", len(val)//batch_size)
    print("Testset size: ", len(testset)//batch_size)

    acc = Metric(name="accuracy", fn=compute_accuracy)

    m.train(
        epochs=50,
        datasets={"train": trainloader, "val": valloader},
        metrics=[acc],
        debug=hparams.debug,
    )

    m.test(
        datasets={"test": testloader},
    )
