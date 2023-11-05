import re
import os
from matplotlib import pyplot as plt

def extract_metrics(input_string):
    #"Epoch: 0 - train loss: 2.2142 accuracy: 15.5854"
    #  val loss: 5.3027 accuracy: 29.3792 best_accuracy: 30.9313
    # 5 train loss: 2.1032 accuracy: 22.6025
    pattern = r"(\d+) train loss: (\d+\.\d+) accuracy: (\d+\.\d+)\n val loss: (\d+\.\d+) accuracy: (\d+\.\d+)"
    matches = re.findall(pattern, input_string)

    # Test loss: 2.0686 Accuracy: 34.69%
    pattern_test = r"Test loss: (\d+\.\d+) Accuracy: (\d+\.\d+)%"
    matches_test = re.findall(pattern_test, input_string)

    extracted_data = {
        "epoch": [],
        "train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for match in matches:
        extracted_data["epoch"].append(int(match[0]))
        extracted_data["train_accuracy"].append(float(match[2]))
        extracted_data["train_loss"].append(float(match[1]))
        extracted_data["val_accuracy"].append(float(match[4]))
        extracted_data["val_loss"].append(float(match[3]))

    print(matches_test)

    extracted_data["test_accuracy"].append(matches_test[0][1])
    extracted_data["test_loss"].append(matches_test[0][0])   

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
    return extracted_data    

if __name__ == "__main__":
    path = "../code/task1/task1.1-2/"    
    fig1, axs1  = plt.subplots(3,3, figsize=(18,12))
    fig1.suptitle("20 Epochs - Batch-size 64")        
    fig2, axs2  = plt.subplots(3,3, figsize=(18,12))
    fig2.suptitle("20 Epochs - Batch-size 128")

    for x in os.listdir(path):
        if("results" in x):

            # create 2 plots             

            full_path = path + x

            extracted_data = plot_folder(full_path)
                
            print(full_path)
            print("adam" in full_path)

            if("64" in full_path):
                if("sgd" in full_path):
                    if("0.001" in full_path):
                        axs1[0,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[0,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[0,0].set_title('lr:0.001 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs1[0,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[0,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[0,1].set_title('lr:0.0001 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs1[0,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[0,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[0,2].set_title('lr:1e-05 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                elif("adamW" in full_path):
                    if("0.001" in full_path):
                        axs1[2,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[2,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[2,0].set_title('lr:0.001 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs1[2,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[2,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[2,1].set_title('lr:0.0001 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs1[2,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[2,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')                    
                        axs1[2,2].set_title('lr:1e-05 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                else:
                    if("0.001" in full_path):
                        axs1[1,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[1,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[1,0].set_title('lr:0.001 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs1[1,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[1,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[1,1].set_title('lr:0.0001 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs1[1,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs1[1,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs1[1,2].set_title('lr:1e-05 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")
                    
            else:
                if("sgd" in full_path):
                    if("0.001" in full_path):
                        axs2[0,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[0,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[0,0].set_title('lr:0.001 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs2[0,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[0,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[0,1].set_title('lr:0.0001 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs2[0,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[0,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[0,2].set_title('lr:1e-05 - optim: SGD - ' + extracted_data["test_accuracy"][0] + "%")
                elif("adamW" in full_path):
                    if("0.001" in full_path):
                        axs2[2,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[2,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[2,0].set_title('lr:0.001 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs2[2,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[2,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[2,1].set_title('lr:0.0001 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs2[2,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[2,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[2,2].set_title('lr:1e-05 - optim: adamW - ' + extracted_data["test_accuracy"][0] + "%")
                else:
                    if("0.001" in full_path):
                        axs2[1,0].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[1,0].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[1,0].set_title('lr:0.001 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")
                    elif("0.0001" in full_path):                        
                        axs2[1,1].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[1,1].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')
                        axs2[1,1].set_title('lr:0.0001 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")
                    else:
                        axs2[1,2].plot(extracted_data["epoch"], extracted_data["train_loss"], alpha=0.7, c='blue', label='Parameters')
                        axs2[1,2].plot(extracted_data["epoch"], extracted_data["val_loss"], alpha=0.7, c='red', label='Parameters')           
                        axs2[1,2].set_title('lr:1e-05 - optim: adam - ' + extracted_data["test_accuracy"][0] + "%")

    plt.show()
    fig1.savefig("./search_64.jpg")
    fig2.savefig("./search_128.jpg")