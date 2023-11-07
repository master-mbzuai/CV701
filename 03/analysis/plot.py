import re
import os
from matplotlib import pyplot as plt

def extract_metrics(input_string):
    #"Epoch: 0 - train loss: 2.2142 accuracy: 15.5854"
    #  val loss: 5.3027 accuracy: 29.3792 best_accuracy: 30.9313
    # 5 train loss: 2.1032 accuracy: 22.6025

    print(input_string)

    # Epoch: 4 - train loss: 2.2950 accuracy: 7.6819
    #  val loss: 2.2939 accuracy: 8.1251 best_loss: 2.2560
    pattern = r"Epoch: (\d+) - train loss: (\d+\.\d+) accuracy: (\d+\.\d+)\n val loss: (\d+\.\d+) accuracy: (\d+\.\d+)"
    #pattern = r"(\d+) train loss: (\d+\.\d+) accuracy: (\d+\.\d+)\n val loss: (\d+\.\d+) accuracy: (\d+\.\d+)"
    matches = re.findall(pattern, input_string)

    extracted_data = {
        "epoch": [],
        "train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": []
    }

    for match in matches:
        extracted_data["epoch"].append(int(match[0]))
        extracted_data["train_accuracy"].append(float(match[2]))
        extracted_data["train_loss"].append(float(match[1]))
        extracted_data["val_accuracy"].append(float(match[4]))
        extracted_data["val_loss"].append(float(match[3]))

    return extracted_data

def plot_folder(path):    

    #open text file in read mode
    #text_file = open("../logs/1000_log.txt", "r")
    text_file = open(path + "/log.txt", "r")
    #text_file = open("../results/adaptive_exp_0/90/exp/train_log.txt", "r")

    # Your input string
    #read whole file to a string
    input_string = text_file.read()

    # Extract metrics
    extracted_data = extract_metrics(input_string)
    print(extracted_data)

    # Create the scatter plot
    plt.plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Train loss')
    plt.plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Val loss')

    plt.title('Train loss (blue) VS Validation loss')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Val loss'])
    plt.grid(True)
    plt.savefig(path + 'trainloss.jpg')
    plt.show()

if __name__ == "__main__":
    path = "../code/task1/task1.4/"
    for x in os.listdir(path):
        if("results" in x):
            full_path = path + x
            print(full_path)
            plot_folder(full_path)
    #x = "results_e100_l0.001_adam_model06_ReLU"
    full_path = path + x
    plot_folder(full_path)
