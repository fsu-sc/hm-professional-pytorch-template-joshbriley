import torch
import torch.nn as nn
from base.base_model import BaseModel 

class DynamicModel(BaseModel):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=2, hidden_units=32, 
                 hidden_activation='relu', output_activation='linear'):
        super(DynamicModel, self).__init__()

        assert 1 <= hidden_layers <= 5, "Hidden layers must be between 1 and 5"
        assert 1 <= hidden_units <= 100, "Hidden units must be between 1 and 100"

        self.activation_fn = self.get_activation(hidden_activation)
        self.output_activation_fn = self.get_activation(output_activation)

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(self.activation_fn)

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.activation_fn)

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))
        if output_activation != 'linear':
            layers.append(self.output_activation_fn)

        self.model = nn.Sequential(*layers)

    def get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        return self.model(x)
