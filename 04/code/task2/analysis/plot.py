
import os
import matplotlib.pyplot as plt

logs = ["fps_log.txt", "fps_log_no_nn.txt", "fps_log_optimized.txt"]

for log_file in logs:

    print(log_file)

    with open(log_file, "r") as f:
        lines = f.readlines()

        fps = [float(line.split(" - ")[1].replace("\n", "")) for line in lines[:175]]

        print(fps)

        plt.plot(fps)

plt.legend(["With NN", "Without NN", "With int8 optimization"])
plt.show()