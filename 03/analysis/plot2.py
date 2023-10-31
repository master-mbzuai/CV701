
import re
from matplotlib import pyplot as pltplot

def parsefile(log_text):

    # Extracting train loss and accuracy
    train_loss_and_acc = re.findall(r"train loss: ([0-9.]+) accuracy: ([0-9.]+)", log_text)
    train_loss_and_acc = [(float(loss), float(acc)) for loss, acc in train_loss_and_acc]

    # Extracting validation loss and accuracy
    val_loss_and_acc = re.findall(r"val loss: ([0-9.]+) accuracy: ([0-9.]+)", log_text)
    val_loss_and_acc = [(float(loss), float(acc)) for loss, acc in val_loss_and_acc]

    return train_loss_and_acc, val_loss_and_acc

    print("Train Loss and Accuracy:", train_loss_and_acc)
    print("Validation Loss and Accuracy:", val_loss_and_acc)

path = "../code/task1.1/"
#open text file in read mode
#text_file = open("../logs/1000_log.txt", "r")


text_file = open(path + "adam.txt", "r")
input_string = text_file.read()

adam_t, adam_v = parsefile(input_string)

# text_file = open(path + "adamW.txt", "r")
# input_string = text_file.read()

# adamW = parsefile(input_string)

# text_file = open(path + "SGD.txt", "r")
# input_string = text_file.read()

# SGD = parsefile(input_string)

adam_t = [x[0]/64 for x in adam_t]
epochs = [x for x in range(len(adam_t))]

print(adam_t)

# Create the scatter plot
plt.plot(adam_t, alpha=0.7, c='blue', label='Parameters')
plt.plot(adam_v, alpha=0.7, c='red', label='Parameters')

plt.title('Train loss (blue) VS Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(path + 'trainloss.jpg')
plt.show()