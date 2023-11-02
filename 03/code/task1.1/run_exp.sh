# Define the lists
lr=("0.001" "0.0001" "0.00001")
opt=("adam" "adamW" "SGD")
batch_size=("64" "128")

# Iterate through all combinations
for learning_rate in "${lr[@]}"; do
  for optimizer in "${opt[@]}"; do
    for size in "${batch_size[@]}"; do              
        xterm -e "python main.py --lr $learning_rate --opt $optimizer --batch_size $size --epochs 20; exit" &      
    done
  done
done
