
import torch

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments

if __name__ == "__main__":  

    hparams = parse_arguments()    

    m = FacialPoints(hparams) 

    m.export("model.tflite", "tflite", (3, 224, 224))