import torch
from torch.utils.data import Dataset
import numpy as np
from ..base.base_data_loader import BaseDataLoader  

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function
        self.x = np.random.uniform(0, 2 * np.pi, n_samples)
        self.y = self.generate_y(self.x, function)

        # Normalize x and y (zero mean, unit std)
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

        self.x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def generate_y(self, x, function):
        epsilon = np.random.uniform(-1, 1, size=x.shape)
        if function == 'linear':
            return 1.5 * x + 0.3 + epsilon
        elif function == 'quadratic':
            return 2 * x**2 + 0.5 * x + 0.3 + epsilon
        elif function == 'harmonic':
            return 0.5 * x**2 + 5 * np.sin(x) + 3 * np.cos(3 * x) + 2 + epsilon
        else:
            raise ValueError(f"Unknown function type: {function}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples=n_samples, function=function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
