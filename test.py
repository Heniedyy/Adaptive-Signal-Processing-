import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support

class UrbanSoundDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.data_dir, file_name)
        try:
            mel_spectrogram = np.load(file_path)
        except ValueError:
            print(f"Error loading: {file_name}")
            return self.__getitem__(np.random.randint(0, self.__len__()))

        # Pad or truncate the mel spectrogram to a fixed size
        fixed_size = (64, 173)
        if mel_spectrogram.shape != fixed_size:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, fixed_size[0] - mel_spectrogram.shape[0]), (0, fixed_size[1] - mel_spectrogram.shape[1])), mode='constant')
            mel_spectrogram = mel_spectrogram[:fixed_size[0], :fixed_size[1]]

        # Extract the label from the filename
        label_str = file_name.split('_')[-1].split('.')[0]
        label_map = {
            'lms': 0,
            'nlms': 1,
            'rls': 2,
            'hybrid': 3,
        }
        label = label_map[label_str]

        return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0), label

class MelSpectrogramModel(nn.Module):
    def __init__(self):
        super(MelSpectrogramModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Remove conv3, bn3, pool3
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 16 * 43, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        # Remove the third convolutional block
        # x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 16 * 43)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    test_data_directory = 'test'  # Specify your test data directory
    batch_size = 32

    test_dataset = UrbanSoundDataset(test_data_directory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MelSpectrogramModel()  # Create an instance of the model
    model.load_state_dict(torch.load('trained_model.pth'))  # Load the model weights

    test_model(model, test_loader)  # Test the model
