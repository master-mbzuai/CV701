
import torch

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments
from micromind import convert


if __name__ == "__main__":  

    hparams = parse_arguments()    

    m = FacialPoints(hparams) 

    m.export("model.tflite", "tflite", (3, 224, 224))

    #convert.convert_to_tflite(m, "tflite", batch_quant=torch.randn(224,224,3))