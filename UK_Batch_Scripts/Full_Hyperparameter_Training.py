import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import sys
from pathlib import Path
functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))

from ML_functions import TensorCatchmentDataset

from ML_Models import LSTM_With_Embedding
from ML_Training import train_model 
import ML_functions

train_non_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/train_non_res_tensor_data.pt')
train_non_res_dataset = TensorCatchmentDataset(train_non_res_tensor_data)

val_non_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/val_non_res_tensor_data.pt')
val_non_res_dataset = TensorCatchmentDataset(val_non_res_tensor_data)

train_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/train_res_tensor_data.pt')
train_res_dataset = TensorCatchmentDataset(train_res_tensor_data)

val_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/val_res_tensor_data.pt')
val_res_dataset = TensorCatchmentDataset(val_res_tensor_data)

# Combine reservoir and non-reservoir datasets
train_full_dataset = ConcatDataset([train_res_dataset, train_non_res_dataset])
val_full_dataset = ConcatDataset([val_res_dataset, val_non_res_dataset])

## Fixed Parameters
BATCH_SIZE = 128
NUM_WORKERS = 1
LSTM_OUTPUT_SIZE = 1
warmup = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


train_full_dataloader = DataLoader(train_full_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True

val_full_dataloader  = DataLoader(val_full_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True

for i, batch in enumerate(train_full_dataloader):
    if i == 0:
        print(f"Static shape: {batch['static'].shape}")
        print(f"Temporal shape: {batch['temporal'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"\nProcessing 100 batches...")
    break

STATIC_INPUT_SIZE = batch['static'].shape[1]
TEMPORAL_INPUT_SIZE =  batch['temporal'].shape[-1]

loss_interval = 500 # Perform optimizer step after this many samples

## Currently Fixed Pararmeters
dropout = 0.3
n_epochs = 4
lstm_h_size = 256

# --- Hyperparameter Search Setup ---
hyperparam_grid = {
    'embedding_output_size': [8, 32],
    'learning_rate': [1e-3, 1e-4],
    # 'lstm_hidden_size': [256],
    'num_epochs': [1,4],
    'lstm_num_layers': [1, 2],
    'embedding_hidden_size': [64]
}

# Other fixed parameters


# Iterate through hyperparameter combinations
for emb_o_size in hyperparam_grid['embedding_output_size']:
    for lr in hyperparam_grid['learning_rate']:
        for n_epochs in hyperparam_grid['num_epochs']:
        # for lstm_h_size in hyperparam_grid['lstm_hidden_size']:
            for lstm_n_layers in hyperparam_grid['lstm_num_layers']:
                for emb_h_size in hyperparam_grid['embedding_hidden_size']:
    
    
                    print(f"\n--- Testing Hyperparameters ---")
                    print(f"  Embedding Output Size: {emb_o_size}")
                    print(f"  Embedding Hidden Size: {emb_h_size}")
                    print(f"  Learning Rate: {lr}")
                    print(f"  LSTM Hidden Size: {lstm_h_size}")
                    print(f"  LSTM Num Layers: {lstm_n_layers}")
    
                    # 1. Initialize Model
                    lstm_input_size = TEMPORAL_INPUT_SIZE + emb_o_size
                    
                    Full_LSTM = LSTM_With_Embedding(
                        static_input_size=STATIC_INPUT_SIZE,
                        embedding_hidden_size= emb_h_size, 
                        embedding_output_size=emb_o_size, 
                        lstm_input_size= lstm_input_size,
                        lstm_hidden_size=lstm_h_size, 
                        lstm_num_layers=lstm_n_layers,
                        lstm_output_size=LSTM_OUTPUT_SIZE,
                        dropout= dropout,
                        bidirectional= False ,
                        Sequence_Target= True
                    )
                    Full_LSTM.to(DEVICE)
    
                    # 2. Define Loss Function and Optimizer
                    criterion = ML_functions.NSELoss()
                    optimizer = optim.Adam(Full_LSTM.parameters(), lr=lr) # Varied
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
                    # 4. Train the model
                    print("  Starting training...")
                    train_model(Full_LSTM, train_full_dataloader, criterion, optimizer, scheduler, n_epochs,
                                warmup, loss_interval, DEVICE, train_mode=True)
                    print("  Training finished.")


                    model_save_path =f'/home/mokr/Reservoir_Modelling/CamelsGB_Models/Full/Full_LSTM_Numlayers{lstm_n_layers}_Embhs{emb_h_size}_Embos{emb_o_size}.pt'

                    torch.save(Full_LSTM, model_save_path)
                    print(f"Model saved to {model_save_path}")


                    # 5. Evaluate on Validation Set     
                    print('Validating')
                    try:
                        train_model(Full_LSTM, val_full_dataloader, criterion, optimizer, scheduler, 1, 
                        warmup, loss_interval, DEVICE, train_mode = False)
                    except Exception as e:
                        print(f"Validation failed: {e}")
    
        
        
        