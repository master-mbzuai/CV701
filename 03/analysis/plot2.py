
import re
from matplotlib import pyplot as plt

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


text_file = open(path + "adamW.txt", "r")
input_string = text_file.read()

adamW_t, adamW_v = parsefile(input_string)

adamW_t = [x[0]/64 for x in adamW_t]
adamW_v = [x[0] for x in adamW_v]
epochs = [x for x in range(len(adamW_t))]

#print(adamW_t)

text_file = open(path + "adam.txt", "r")
input_string = text_file.read()

adam_t, adam_v = parsefile(input_string)

adam_t = [x[0] for x in adam_t]
adam_v = [x[0] for x in adam_v]

text_file = open(path + "SGD.txt", "r")
input_string = text_file.read()

SGD_t, SGD_v = parsefile(input_string)

SGD_t = [x[0]/64 for x in SGD_t]
SGD_v = [x[0] for x in SGD_v]

# Create a figure
plt.figure(figsize=(18, 6))

# First subplot
plt.subplot(1, 3, 1)
plt.plot(epochs, SGD_t, alpha=0.7, c='blue', label='Parameters')
plt.plot(epochs, SGD_v, alpha=0.7, c='red', label='Parameters')
plt.grid(True)
plt.title('Train loss (blue) VS Validation loss - SGD ')

# Second subplot
plt.subplot(1, 3, 3)
plt.plot(epochs, adamW_t, alpha=0.7, c='blue', label='Parameters')
plt.plot(epochs, adamW_v, alpha=0.7, c='red', label='Parameters')
plt.grid(True)
plt.title('Train loss (blue) VS Validation loss - ADAMW ')

plt.subplot(1, 3, 2)

# Create the scatter plot
plt.plot(epochs, adam_t, alpha=0.7, c='blue', label='Parameters')
plt.plot(epochs, adam_v, alpha=0.7, c='red', label='Parameters')

plt.title('Train loss (blue) VS Validation loss - ADAM')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(path + 'trainloss.jpg')
plt.show()