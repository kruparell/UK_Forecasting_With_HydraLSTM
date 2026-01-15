import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
from pathlib import Path
functions_path = Path("/home/mokr/Reservoir_Modelling/ML_Functions/")
sys.path.append(str(functions_path))

from ML_functions import TensorCatchmentDataset

import ML_Models
from ML_Models import LSTM_With_Embedding, LSTMModel, HydraModel
from ML_Training import train_model

from ML_functions import NSELoss

# Create dataloaders
train_non_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/train_non_res_tensor_data.pt')
train_non_res_dataset = TensorCatchmentDataset(train_non_res_tensor_data)

val_non_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/val_non_res_tensor_data.pt')
val_non_res_dataset = TensorCatchmentDataset(val_non_res_tensor_data)

train_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/train_res_tensor_data.pt')
train_res_dataset = TensorCatchmentDataset(train_res_tensor_data)

val_res_tensor_data = torch.load('/perm/mokr/CamelsGB_V2/val_res_tensor_data.pt')
val_res_dataset = TensorCatchmentDataset(val_res_tensor_data)

## Fixed Parameters
BATCH_SIZE = 128
NUM_WORKERS = 1
LSTM_OUTPUT_SIZE = 1
GAMMA = 0.95
loss_interval = 500 # Perform optimizer step after this many samples
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


train_reservoir_dataloader = DataLoader(train_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True

val_reservoir_dataloader  = DataLoader(val_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True


train_non_reservoir_dataloader = DataLoader(train_non_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True

val_non_reservoir_dataloader  = DataLoader(val_non_res_dataset, batch_size= BATCH_SIZE, shuffle=True, 
                              num_workers= NUM_WORKERS) # , pin_memory=True, persistent_workers=True


for i, batch in enumerate(train_reservoir_dataloader):
    if i == 0:
        print(f"Static shape: {batch['static'].shape}")
        print(f"Temporal shape: {batch['temporal'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"\nProcessing 100 batches...")
    break

STATIC_INPUT_SIZE = batch['static'].shape[1]
TEMPORAL_INPUT_SIZE =  batch['temporal'].shape[-1]
## Currently Fixed Pararmeters
dropout = 0.3
num_epochs = 4
lstm_n_layers = 1
warmup = 5
lstm_h_size = 256

# --- Hyperparameter Search Setup ---
hyperparam_grid = {
    'embedding_output_size': [16],
    'learning_rate': [1e-3],
    # 'lstm_hidden_size': [256],
    'num_epochs': [2],
    # 'lstm_num_layers': [1],
    'hydra_body_output_size': [16],
    'embedding_hidden_size': [64]

}



# Iterate through hyperparameter combinations
for emb_o_size in hyperparam_grid['embedding_output_size']:
    for lr in hyperparam_grid['learning_rate']:
        for n_epochs in hyperparam_grid['num_epochs']:
        # for lstm_h_size in hyperparam_grid['lstm_hidden_size']:
            # for lstm_n_layers in hyperparam_grid['lstm_num_layers']:
            for hydra_bo_size in hyperparam_grid['hydra_body_output_size']:
                for emb_h_size in hyperparam_grid['embedding_hidden_size']:
    
    
                    print(f"\n--- Testing Hyperparameters ---")
                    print(f"  Embedding Output Size: {emb_o_size}")
                    print(f"  Embedding Hidden Size: {emb_h_size}")
                    print(f"  Learning Rate: {lr}")
                    # print(f"  LSTM Hidden Size: {lstm_h_size}")
                    print(f'Number of Epochs:, {n_epochs}')         
                    print(f"  Hydra Body Output Size: {hydra_bo_size}")
    
                    # 1. Initialize Model
                    body_input_size = TEMPORAL_INPUT_SIZE + emb_o_size
                    
                    Hydra_Body = LSTM_With_Embedding(
                        static_input_size=STATIC_INPUT_SIZE, 
                        embedding_hidden_size=emb_h_size,  
                        embedding_output_size=emb_o_size,  
                        lstm_input_size=body_input_size, 
                        lstm_hidden_size=lstm_h_size,  
                        lstm_num_layers=lstm_n_layers, 
                        lstm_output_size=hydra_bo_size,
                        dropout=dropout, 
                        bidirectional=False, 
                        Sequence_Target=True
                    )

                    Hydra_Head = LSTMModel(
                        input_size=hydra_bo_size, 
                        hidden_size=round(lstm_h_size/2), 
                        num_layers= lstm_n_layers,
                        output_size=1, 
                        dropout=dropout, 
                        bidirectional=False, 
                        Sequence_Target=True
                    )                
                            
                    
                    Hydra_Res_Head_Embedding = LSTM_With_Embedding(
                        static_input_size=STATIC_INPUT_SIZE, 
                        embedding_hidden_size=emb_h_size,  
                        embedding_output_size=emb_o_size,  
                        lstm_input_size=hydra_bo_size + emb_o_size, 
                        lstm_hidden_size=round(lstm_h_size/2), 
                        lstm_num_layers= lstm_n_layers,
                        lstm_output_size=1,
                        dropout=dropout,  
                        bidirectional=False, 
                        Sequence_Target=True
                    )
                    
                    Hydra_Main = HydraModel(Hydra_Body, Hydra_Head)
                    
                    Hydra_Main.to(DEVICE)
                    Hydra_Head.to(DEVICE)
                    Hydra_Res_Head_Embedding.to(DEVICE)

                    optimizer_main = optim.Adam(
                        list(Hydra_Main.body.parameters()) + list(Hydra_Main.head_default.parameters()),
                        lr=lr)

                    # Optimizer for ONLY the Reservoir Head
                    optimizer_res_embedding_head = optim.Adam(Hydra_Res_Head_Embedding.parameters(), lr= lr)

                    
                    # Schedulers for each optimizer
                    scheduler_main = optim.lr_scheduler.ExponentialLR(optimizer_main, gamma= GAMMA)
                    scheduler_res_embedding_head = optim.lr_scheduler.ExponentialLR(optimizer_res_embedding_head, gamma= GAMMA)

                    # 2. Define Loss Function and Optimizer
                    criterion = NSELoss()
    
                    # 4. Train the model
                    print("  Starting training Hydra_Main...")
                    train_model(Hydra_Main, train_non_reservoir_dataloader, criterion, optimizer_main, scheduler_main, n_epochs,
                                warmup, loss_interval, DEVICE, train_mode=True)
                    
                    print("  Training finished.")


                    # 5. Evaluate on Validation Set     
                    print('Validating Hydra_Main')
                    try:
                        train_model(Hydra_Main, val_non_reservoir_dataloader, criterion, optimizer_main, scheduler_main, 1, 
                        warmup, loss_interval, DEVICE, train_mode = False)
                    except Exception as e:
                        print(f"Validation failed: {e}")

                    
                    print("  Starting training Hydra_Head...")
                    train_model([Hydra_Body, Hydra_Res_Head_Embedding], train_reservoir_dataloader, criterion, optimizer_res_embedding_head, scheduler_res_embedding_head, n_epochs,
                                warmup, loss_interval, DEVICE, train_mode=True)
                    print("  Training finished.")
                    
                    model_save_path = f'/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/Hydra_LSTM_Numlayers{lstm_n_layers}_Embhs{emb_h_size}_Embos{emb_o_size}_BodyOut{hydra_bo_size}.pt'

                    save_dir = '/home/mokr/Reservoir_Modelling/CamelsGB_Models/Hydra/'
                    base_name = f'Hydra_LSTM_Numlayers{lstm_n_layers}_Embhs{emb_h_size}_Embos{emb_o_size}_BodyOut{hydra_bo_size}'
                    
                    # 2. Define specific paths for all 3 components
                    main_path = f"{save_dir}{base_name}_Body.pt"
                    head_path = f"{save_dir}{base_name}_Main_Head.pt"
                    res_emb_path = f"{save_dir}{base_name}_Res_Embedding_Head.pt"
                    
                    # 3. Save the state_dicts
                    torch.save(Hydra_Body, main_path)
                    torch.save(Hydra_Head, head_path)
                    torch.save(Hydra_Res_Head_Embedding, res_emb_path)
                    
                    print(f"Model saved to {model_save_path}")


                    # 5. Evaluate on Validation Set     
                    print('Validating')
                    try:
                        train_model(Hydra_Main, val_reservoir_dataloader, criterion, optimizer_res_embedding_head, scheduler_res_embedding_head, 1, 
                        warmup, loss_interval, DEVICE, train_mode = False)
                    except Exception as e:
                        print(f"Validation failed: {e}")
    
        
        