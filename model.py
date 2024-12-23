import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for the MNIST classification.
    Used by both the target and shadow models.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
 
        self.fc = nn.Sequential(
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AttackMLP(nn.Module):
    """
    Enhanced MLP for the attack model with multiple hidden layers,
    batch normalization, and dropout for better performance.
    
    Input: Probability vector of size 10 (from softmax outputs)
    Output: Single probability (0=non-member, 1=member)
    """
    def __init__(self, input_size=10, hidden_sizes=[128, 64, 32], dropout_prob=0.5):
        super(AttackMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

