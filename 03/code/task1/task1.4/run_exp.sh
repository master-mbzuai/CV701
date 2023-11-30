# Define the lists
opt=("01" "02" "03" "04" "05" "06")

# Iterate through all combinations
for optimizer in "${opt[@]}"; do  
  xterm -e "python main.py --opt adam --model_name model$optimizer --epochs 50; exit" &  
done
