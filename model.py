import torch
import torch.nn as nn
import torch.nn.functional as F

class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        # Input shape: (Batch Size, 1, 64, 128)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 3 maxpools of 2x2, spatial dimensions will be divided by 8.
        # 64 / 8 = 8, 128 / 8 = 16
        # So flattened feature size will be 64 * 8 * 16 = 8192
        
        self.fc1 = nn.Linear(64 * 8 * 16, 128)
        self.fc2 = nn.Linear(128, 2) # 2 classes: male(0) and female(1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
