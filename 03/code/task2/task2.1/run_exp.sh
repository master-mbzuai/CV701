# Define the lists
opt=("sgd" "adam" "adamW" "adadelta" "nadam")

# Iterate through all combinations
for optimizer in "${opt[@]}"; do  
  xterm -e "python main.py --opt $optimizer --model_name model01 --epochs 50; exit" &  
done
