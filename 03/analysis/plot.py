import re
from matplotlib import pyplot as plt

def extract_metrics(input_string):
    #"Epoch: 0 - train loss: 2.2142 accuracy: 15.5854"
    pattern = r"Epoch: (\d+) - train loss: (\d+\.\d+) accuracy: (\d+\.\d+)\n val loss: (\d+\.\d+) accuracy: (\d+\.\d+)"
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


path = "../code/task1.3/results_e20_l0.001_adam_model01_SiLU/"

#open text file in read mode
#text_file = open("../logs/1000_log.txt", "r")
text_file = open(path + "log.txt", "r")
#text_file = open("../results/adaptive_exp_0/90/exp/train_log.txt", "r")

# Your input string
#read whole file to a string
input_string = text_file.read()

# Extract metrics
extracted_data = extract_metrics(input_string)
print(extracted_data)

# Create the scatter plot
plt.plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
plt.plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')

plt.title('Train loss (blue) VS Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(path + 'trainloss.jpg')
plt.show()