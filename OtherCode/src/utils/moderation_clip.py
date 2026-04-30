# Define ModerationModel as before
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, logits, labels):
        bce = self.bce_loss(logits, labels)
        mse = self.mse_loss(torch.sigmoid(logits), labels)  # Apply sigmoid for MSE calculation
        return self.bce_weight * bce + self.mse_weight * mse

class ModerationModel(nn.Module):
    def __init__(self, model_name):
        if model_name == "ViT-bigG-14":
            input_size = 1280
            hidden_size = 256
            output_size = 11
        else:
            input_size = 768
            hidden_size = 256
            output_size = 11
        super(ModerationModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        return x


# Updated getEmbeddings function as before

# Custom Dataset Class
class ModerationDataset(Dataset):
    def __init__(self, data, num_labels=11):
        self.embeddings = data[:, :-num_labels]  # Split embeddings
        self.labels = data[:, -num_labels:]  # Split labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return embeddings, labels