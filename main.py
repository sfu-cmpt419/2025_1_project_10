import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from src.preprocess import *

train_df = pd.read_csv('data/train_df.csv', usecols=['image_path', 'Target'])

train_df['Target'] = train_df['Target'].str.strip()
train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join("data", x.lstrip("./")))

split_data = train_df[['image_path', 'Target']]
train, test = train_test_split(train_df[['image_path', 'Target']], test_size=0.2)
train.shape, test.shape

multi_labels = [i for i, target in enumerate(train_df['Target']) if len(target) > 2]

corrected_labels = []

for ml in multi_labels:    
    corrected_labels.append([train_df.loc[ml, 'image_path'], train_df.loc[ml, 'Target'].split(' ')[0]])
    
train_df = pd.concat([train_df.drop(train_df.loc[multi_labels].index),
                     pd.DataFrame(corrected_labels, columns=['image_path', 'Target'])], ignore_index=True)


# Step 1: Dataset and DataLoader
class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns "Target" and "image_path".
            transform (callable, optional): Transform to be applied on an image.
        """
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = int(self.df.iloc[idx]['Target'])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img_tensor = pre_processing(img)
        return img_tensor, label


train_dataset = XrayDataset(train)  # Don't pass transform
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = XrayDataset(test)    # Same here
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Step 2: CNN Model Definition
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=16):
        """
        Args:
            num_classes (int): Number of target classes.
        """
        super(CNNClassifier, self).__init__()
        # Using padding to keep spatial dimensions, then reducing with pooling.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive pooling to get a fixed feature size regardless of the input dimensions.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 56, 56]
        x = self.avgpool(x)                   # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)             # flatten to [batch, 64*7*7]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                     # logits output
        return x

# Determine the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# It's a good idea to automatically determine the number of classes.
num_classes = train_df["Target"].nunique()  
model = CNNClassifier(num_classes=num_classes).to(device)

# Step 3: Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()

# Add weight decay to help generalization
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Step 4: Training Loop
epochs = 10  # You can adjust the number of epochs as needed

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        # Ensure labels are the right type (long) for CrossEntropyLoss.
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Optional: Save the trained model
torch.save(model.state_dict(), "xray_model.pth")


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# First, accumulate predictions and true labels from the test dataset.
true_labels = []
pred_labels = []

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        # Ensure labels are tensors and in the correct type
        labels = torch.tensor(labels, device=device, dtype=torch.long)
        
        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Save predictions and true labels
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Compute the confusion matrix from the true and predicted labels.
conf_metric = confusion_matrix(true_labels, pred_labels)

# Normalize the confusion matrix by dividing each row by its sum (i.e., relative percentages).
conf_metric_normalized = conf_metric / np.sum(conf_metric, axis=1, keepdims=True)

# Plot the confusion matrix using Seaborn.
plt.figure(figsize=[12,12], dpi=100)
sns.heatmap(np.round(conf_metric_normalized, 2),
            cbar=False,
            annot=True,
            annot_kws={"size": 9},
            cmap=plt.cm.Blues)
plt.xlabel('True labels')
plt.ylabel('Predicted labels')
plt.title('Normalized Confusion Matrix')
plt.show()