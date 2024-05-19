import torch
import torch.nn as nn
import numpy as np

class MelSpectrogramModel(nn.Module):
    def __init__(self):
        super(MelSpectrogramModel, self).__init__()
        # Assuming a different model architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Temporarily instantiate a dummy input to calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 173)
            dummy_output = self.forward_features(dummy_input)
            self.flattened_size = int(np.prod(dummy_output.size()))

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward_features(self, x):
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def print_model_details(model):
    print("Model architecture:\n", model)
    trainable_params = sum(p.numel() for p in model.parameters())
    non_trainable_params = 0
    total_params = trainable_params
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

if __name__ == "__main__":
    model = MelSpectrogramModel()
    print_model_details(model)