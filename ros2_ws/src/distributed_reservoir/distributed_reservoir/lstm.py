import torch
import torch.nn as nn

# ---- LSTM Model Definition (without Defaults!) ----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, T, input_size]
        out, _ = self.lstm(x)
        # last time step
        return self.fc(out[:, -1, :])
    
    def process_input(self, input_data, **kwargs):
        '''
        Process input data through the LSTM.
        Standard interface for edge server integration.
        Automatically handles input shape transformation.
        
        Args:
            input_data: Tensor of shape (batch_size, input_dim)
            **kwargs: Additional arguments (unused, for interface compatibility)
        
        Returns:
            LSTM output tensor of shape (batch_size, output_size)
        '''
        # LSTM expects (batch_size, sequence_length, input_size)
        # input_data is (batch_size, input_dim), treat as sequence of length 1
        if input_data.dim() == 2:
            input_data = input_data.unsqueeze(1)  # (batch_size, 1, input_dim)
        return self.forward(input_data)