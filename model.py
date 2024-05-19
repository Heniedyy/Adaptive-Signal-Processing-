import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle, product
from sklearn.preprocessing import label_binarize

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

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        # Extract the label from the filename, assuming it is after the last underscore and before ".npy"
        label_str = file_name.split('_')[-1].split('.')[0]

        # Map the label strings to integers
        label_map = {
            'lms': 0,
            'nlms': 1,
            'rls': 2,
            'hybrid': 3,
        }
        label = label_map[label_str]

        return mel_spectrogram, label

class MelSpectrogramModel(nn.Module):
    def __init__(self):
        super(MelSpectrogramModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 43, 128)
        self.fc2 = nn.Linear(128, 4)  # Assuming 4 classes

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 43)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(y_true, y_score, n_classes, title='ROC Curve'):
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def train(model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            inputs = inputs.float()
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_accuracy = 0.0
        total = 0
        true_labels = []
        predicted_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.float()
                labels = labels.long()

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                val_accuracy += (predicted == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        val_accuracy = val_accuracy / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro', zero_division=1
        )

        # Print validation metrics in a table
        print(f"| {'Epoch':^6} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10} |")
        print(f"|{'-' * 6}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|")
        print(f"| {epoch + 1:^6} | {val_accuracy:^10.4f} | {precision:^10.4f} | {recall:^10.4f} | {f1_score:^10.4f} |")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, classes=['lms', 'nlms', 'rls', 'hybrid'])

    # ROC Curve
    plot_roc_curve(true_labels, np.array(all_probs), n_classes=4)

    return model


if __name__ == "__main__":
    data_dir = "model"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # Create the dataset and DataLoader
    dataset = UrbanSoundDataset(data_dir)

    # Ensure equal number of files per class in training and validation
    class_indices = {0: [], 1: [], 2: [], 3: []}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    for indices in class_indices.values():
        np.random.shuffle(indices)
        split_point = int(0.8 * len(indices))
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create the model
    model = MelSpectrogramModel()

    # Train the model
    trained_model = train(model, train_loader, val_loader, num_epochs, learning_rate)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'trained_model.pth')
