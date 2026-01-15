import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


from collections import defaultdict




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=False, Sequence_Target=False):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.No_Directions = 2 if bidirectional else 1
        self.Sequence_Target = Sequence_Target
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, 
                             dropout=dropout, 
                             bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size)
    def forward(self, x):
        # Correct initialization of hidden state for batched input
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.No_Directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.No_Directions, batch_size, self.hidden_size).to(x.device)
        torch.nn.init.xavier_uniform_(h0)
        torch.nn.init.xavier_uniform_(c0)
        
        # Pass initial states to LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout
        out = self.dropout(out)
        
        # Handle sequence target or last time step
        if not self.Sequence_Target:
            # Take the last time step for each sequence in the batch
            out = out[:, -1, :]  # Shape: [batch_size, hidden_size * No_Directions]
        
        # Final linear layer
        out = self.fc(out)
        return out
        
class Static_Embedding(nn.Module):
    """A simple MLP to create an embedding from static features."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class LSTM_With_Embedding(nn.Module):
    # Maybe edit this to have a temporal embedding too
    def __init__(self, static_input_size, embedding_hidden_size, embedding_output_size, 
                 lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size,
                 dropout=0.0, bidirectional=False, Sequence_Target=False):
        super(LSTM_With_Embedding, self).__init__()
        
        self.static_embedding = Static_Embedding(static_input_size, embedding_hidden_size, embedding_output_size)
        self.lstm = LSTMModel(lstm_input_size, lstm_hidden_size, lstm_num_layers, 
                             lstm_output_size, dropout, bidirectional, Sequence_Target)
        
        
    def forward(self, static_input, temporal_input):
        # static_input shape: [batch_size, static_input_size]
        batch_size = static_input.size(0)
        
        # Get static embedding
        embedding = self.static_embedding(static_input)  # [batch_size, embedding_output_size]
        
        # Repeat embedding across time steps to create unchanging time series
        # Shape: [batch_size, sequence_length, embedding_output_size]

        sequence_length = temporal_input.size(1)
        static_input = embedding.unsqueeze(1).repeat(1, sequence_length, 1)

        full_input = torch.cat([temporal_input, static_input], dim=2)
        # Feed to LSTM
        output = self.lstm(full_input)
        
        return output


class Hindcast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Hindcast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
        
        # h0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)

        if len(np.shape(x)) == 3:
            h0 = torch.zeros( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = torch.zeros( self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * self.No_Directions, self.hidden_size).to(x.device)
            
        out, (hn, cn) = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out, hn, cn

class Forecast_LSTM_Block(nn.Module):
    # This block serves to take in historic data and output the initial memory and hidden 
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.0, bidirectional = False):
        super(Forecast_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2

    def forward(self, x, h0, c0):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
   
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out
    
class Google_LSTMModel(nn.Module):
  def __init__(self, hindcast):
    super(Google_LSTMModel, self).__init__()
    self.hindcast = hindcast
    # self.forecast = forecast
    
  def forward(self, history):
    
    # get states from hindcast model
    # need to decide whether the head recieves the raw history or an encoding of it
    hind_out, hn,cn = self.hindcast(history)
    
    # get forecasts from forecast model
    #out = self.forecast(forecasts, hn,cn)
    return hind_out #, hind_out



class Hydra_LSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, H0_sequences_size = int(90*10/7), dropout = 0.0, bidirectional = False):
        super(Hydra_LSTM_Block, self).__init__()
        self.bidirectional = bidirectional  # Store bidirectional as an instance variable
        self.No_Directions = 1 if not bidirectional else 2
        self.H0_sequences_size = H0_sequences_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2
        # Linear layers for mapping H0_sequences and H0_static to h0 and c0 sizes
        self.linear_h0 = nn.Linear(H0_sequences_size, self.num_layers * self.No_Directions * self.hidden_size)


    def forward(self, x, H0_sequences):
        # Map H0_sequences and H0_static to the appropriate sizes
        # Is this implementation of history doing anything
        
        h0 = self.linear_h0(H0_sequences)
        if len(np.shape(x)) == 3:
            h0 = h0.view( self.num_layers * self.No_Directions, x.size(0), self.hidden_size).to(x.device)
        else:
            h0 = h0.view( self.num_layers * self.No_Directions, self.hidden_size).to(x.device)

        c0 = h0.clone()
       
        out, _ = self.lstm(x, (h0, c0)) 
        out = self.dropout(out)
        out = self.fc(out)  # Take the output from the last time step
        return out


# --- Main Body Block ---

class HydraBody(nn.Module):
    """
    Processes and combines static and temporal data.

    This block takes static features, creates an embedding, and concatenates
    it with each time step of the temporal features. The result is then
    processed by an LSTM.
    """
    def __init__(self, static_input_size, embedding_hidden_size, embedding_output_size,
                 temporal_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size,
                 dropout=0.0, bidirectional=False, sequence_target = True):
        super().__init__()

        # Sub-module for processing static data
        self.static_embedding = Static_Embedding(
            static_input_size,
            embedding_hidden_size,
            embedding_output_size
        )

        # The combined input size for the LSTM is the sum of temporal features
        # and the static embedding features.
        lstm_input_size = temporal_input_size + embedding_output_size
        # Sub-module for processing the combined sequence
        self.sequential_processor = LSTMModel(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_size= lstm_output_size,
            dropout=dropout,
            bidirectional=bidirectional,
            Sequence_Target= sequence_target
        )

    def forward(self, static_input, temporal_input):
        """
        Forward pass for the model body.

        Args:
            static_input (torch.Tensor): Shape [batch_size, static_input_size]
            temporal_input (torch.Tensor): Shape [batch_size, sequence_length, temporal_input_size]

        Returns:
            torch.Tensor: LSTM output of shape [batch_size, sequence_length, lstm_hidden_size * num_directions]
        """
        # 1. Process static input to get an embedding
        # Shape: [batch_size, embedding_output_size]
        static_embedded = self.static_embedding(static_input)

        # 2. Expand the static embedding to match the temporal sequence length
        sequence_length = temporal_input.size(1)
        # Shape becomes: [batch_size, 1, embedding_output_size]
        static_expanded = static_embedded.unsqueeze(1)
        # Shape becomes: [batch_size, sequence_length, embedding_output_size]
        static_repeated = static_expanded.repeat(1, sequence_length, 1)

        # 3. Concatenate temporal input with the repeated static features
        # Shape: [batch_size, sequence_length, temporal_input_size + embedding_output_size]
        combined_input = torch.cat([temporal_input, static_repeated], dim=2)

        # 4. Pass the combined sequence through the LSTM
        sequential_output = self.sequential_processor(combined_input)

        return sequential_output

class HydraHead(nn.Module):
    """
    Processes the output sequence from the Body to produce the final prediction.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, bidirectional, sequence_target=False):
        """
        Args:
            input_size (int): The feature dimension of the input sequence (e.g., lstm_hidden_size).
            hidden_size (int): The size of the intermediate hidden layer.
            output_size (int): The final output dimension for each time step.
            sequence_target (bool): If True, output the full sequence. If False,
                                    output only the prediction from the last time step.
        """
        super().__init__()
        self.sequence_target = sequence_target

        self.lstm = nn.LSTM(input_size, hidden_size, outputs_size, num_layers, dropout= dropout, bidirectional = bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * self.No_Directions, output_size) # If bidirectional is true need the *2
        # Linear layers for mapping H0_sequences and H0_static to h0 and c0 sizes
        self.linear_h0 = nn.Linear(H0_sequences_size, self.num_layers * self.No_Directions * self.hidden_size)
        
    def forward(self, x):
        """
        Forward pass for the model head.

        Args:
            x (torch.Tensor): The output from the HydraBody.
                              Shape: [batch_size, sequence_length, body_output_size]

        Returns:
            torch.Tensor: The final model prediction.
                          Shape: [batch_size, sequence_length, output_size] if sequence_target is True.
                          Shape: [batch_size, output_size] if sequence_target is False.
        """

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        if self.sequence_target:
            # Apply to every time step for a sequence-to-sequence output
            prediction = self.output_layer(lstm_out)
        else:
            # Take only the output from the last time step for a sequence-to-one output
            last_time_step_output = lstm_out[:, -1, :]
            prediction = self.output_layer(last_time_step_output)

        return prediction
        return output


class HydraModel(nn.Module):
    def __init__(self, body, head_default):
        super().__init__()
        self.body = body
        self.head_default = head_default

    def forward(self, static_input, temporal_input):
        """
        Forward pass that directs data to the correct head based on the mode.
        """
        # Get the shared output from the body
        body_output = self.body(static_input, temporal_input)

        return self.head_default(body_output)
        
class CombinedHydraModel(nn.Module):
    def __init__(self, body_config, head_config):
        super().__init__()
        # In a Hydra setup, you'd pass config sub-groups here
        self.body = HydraBody(**body_config)
        self.head = HydraHead(**head_config)

    def forward(self, static_input, temporal_input):
        # Data flows from body to head
        body_output = self.body(static_input, temporal_input)
        final_output = self.head(body_output)
        return final_output
        
# Do we want hindcast and forecast num-layers to be different?
def define_models(hindcast_input_size, forecast_input_size, hidden_size, num_layers, dropout, bidirectional, learning_rate, copies = 3, forecast_output_size = 3, device = 'cpu'):
    models = {}
    params_to_optimize = {}
    optimizers = {}
    schedulers = {}
    
    hindcast_output_size = forecast_output_size
    for copy in range(copies):
        models[copy] = Google_Model_Block(hindcast_input_size, forecast_input_size, hindcast_output_size, forecast_output_size, hidden_size, num_layers, device, dropout, bidirectional)
        
        models[copy].to(device)
        params_to_optimize[copy] = list(models[copy].parameters())
        # Probably should be doing 1e-2 and 10
        optimizers[copy] = torch.optim.Adam(params_to_optimize[copy], lr= learning_rate, weight_decay = 1e-4)
        schedulers[copy] = lr_scheduler.StepLR(optimizers[copy], step_size= 5000, gamma=0.95) #.CosineAnnealingLR(optimizers[copy], T_max = 100000,)

        

    return models, params_to_optimize, optimizers, schedulers

# Model Running Code
# Key difference here is that we won't need to restrict the history length to 90, it can be variable


