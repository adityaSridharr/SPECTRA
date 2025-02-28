import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
import numpy as np


args = {
    'cuda': True,
    'num_gpus': 1,
    'seed': 2018,
    'train': True,
    'test': False,
    'load_model': True,
    'model_path': 'models',
    'results_path': 'out',
    'log_path': 'log',
    'summary_path': 'results/summary.csv',
    'cuda': True,
    'num_gpus': 2,
    'seed': 2018,
    'h_type': 'cnn',
    'concept_dim': 1,
    'nconcepts': 10,
    'h_sparsity': 1e-4,
    'nobias': False,
    'positive_theta': False,
    'theta_arch': 'simple',
    'theta_dim': -1,
    'theta_reg_type': 'grad3',
    'theta_reg_lambda': 1e-2,
    'opt': 'adam',
    'lr': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'objective': 'cross_entropy',
    'dropout': 0.0,
    'weight_decay': 1e-3,
    'dataset': 'pathology',
    'embedding': 'pathology',
    'nclasses': 10,
    'num_workers': 0,
    'print_freq': 200,
    'debug': False
}


def load_data(args):
    transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
# import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the directory exists
model_save_path = 'models/classifier_weights_new_train.pth'
image_save_path = 'images/classifier_images_new_train.png'
model_save_path_interpretable = 'interpretable_classifier_weights.pth'

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input channels = 3 (RGB), Output channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output channels = 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output channels = 128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output channels = 256

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected (dense) layer
        self.fc = nn.Linear(128 * 2 * 2, 10)  # CIFAR-10 has 10 classes, 256 channels with 2x2 spatial size after pooling
        
    def forward(self, x):
        # Apply convolutions and max-pooling
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 32, 32) -> (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 16, 16) -> (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 64, 8, 8) -> (B, 128, 4, 4)
        x = self.pool(F.relu(self.conv4(x)))  # (B, 128, 4, 4) -> (B, 256, 2, 2)

        # Flatten the output before the classification layer
        x_flat = x.view(x.size(0), -1)  # Flatten the tensor to (B, 256 * 2 * 2)
        
        # Classification output
        class_output = self.fc(x_flat)  # (B, 10) - 10 classes
        
        # Return the output before the classification layer (conv4 output) and the classification output
        return x, class_output

class InterpretableClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(InterpretableClassifier, self).__init__()
        
        # Define convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 4, 4]
            nn.ReLU(),

        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # [B, 16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),    # [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),                        # [B, 3, 32, 32]
            nn.Sigmoid()  # Normalize the output
        )

    def forward(self, x):
        features = self.features(x)  # (batch_size, 128, 2, 2)
        classification_output = self.classifier(features)
        reconstruction_output = self.decoder(features)
        return features, classification_output, reconstruction_output
    

# Example usage:
# input_tensor = torch.randn(16, 3, 32, 32)  # Batch of 16 CIFAR-10 images
# last_conv_output, class_output = model(input_tensor)

from tqdm import tqdm
# Convert args into a dictionary storring all arguments
def train():
    model = Classifier()  # Move model to GPU if available
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 2: Load the data
    train_loader, test_loader, train_dataset, test_dataset = load_data(args)

    # Step 3: Training the model
    epochs = 20
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', dynamic_ncols=True)


        for inputs, labels in pbar:
            inputs, labels = inputs, labels

            optimizer.zero_grad()

            # Forward pass
            _, outputs, reconstruction = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels) + F.mse_loss(reconstruction, inputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=100 * correct / total)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), model_save_path_interpretable)

    # Step 4: Testing the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs, labels
            _, outputs, reconstruction = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


# Training function
def train_interpretable(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Initialize tqdm progress bar for the training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        running_class_loss = 0.0
        running_reconstruction_loss = 0.0
        running_sparsity_loss = 0.0
        for inputs, labels in pbar:
            optimizer.zero_grad()
            _, outputs, reconstruction = model(inputs)
            loss = criterion(outputs, labels) + F.mse_loss(reconstruction, inputs) + 1e-3 * torch.norm(model.features[0].weight, p=1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_class_loss += criterion(outputs, labels).item()
            running_reconstruction_loss += F.mse_loss(reconstruction, inputs).item()
            running_sparsity_loss += 1e-3 * torch.norm(model.features[0].weight, p=1).item()
            
            # Update progress bar description with the current loss
            pbar.set_postfix({"Loss": f"{running_loss / (pbar.n + 1):.4f}"})
        
        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Classification Loss: {running_class_loss / len(train_loader):.4f}, Reconstruction Loss: {running_reconstruction_loss / len(train_loader):.4f}, Sparsity Loss: {running_sparsity_loss / len(train_loader):.4f}")
    
    # Save model weights
    torch.save(model.state_dict(), model_save_path_interpretable)

def evaluate_interpretable(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            _, outputs, reconstruction = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")



import random

def visualize_gradients(model, dataset, selected_features, image_index=0, top_pixels=200):
    # Access the specified image directly from the dataset
    inputs, label = dataset[image_index]
    inputs = inputs.unsqueeze(0).requires_grad_(True)  # Add batch dimension and enable gradient computation

    # Forward pass to get features and classification
    features, outputs, reconstruction = model(inputs)
    
    # Initialize a larger plot to show the original image and its gradient maps
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Gradients of 10 Selected Features w.r.t Image at index {image_index}', fontsize=18)
    
    # Original image plot
    img = inputs[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC
    img = (img * 0.5) + 0.5  # De-normalize
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Remove any remaining slots in the first row (original image row)
    for j in range(1, 5):
        axes[0, j].axis("off")
    
    # Calculate and plot gradients for each selected feature
    for i, feature_idx in enumerate(selected_features):
        # Zero existing gradients
        model.zero_grad()
        
        # Compute the target feature (select feature channel at index `feature_idx`)
        target_feature = features[0, feature_idx, :, :].mean()  # Mean over the 2x2 spatial dimensions
        
        # Backward pass to compute gradients of the target feature w.r.t the input image
        target_feature.backward(retain_graph=True)  # Retain graph to calculate gradients for multiple features
        gradients = inputs.grad.data[0].cpu().numpy()  # Shape: (3, 32, 32)
        
        # Average the gradients across color channels to get a single saliency map
        gradients = np.mean(gradients, axis=0)  # Shape: (32, 32)
        
        # Normalize the gradients for visualization
        gradients = (gradients - gradients.min()) / (gradients.max())
        
        
        # Highlight the top `top_pixels` pixels in the original image
        flattened_grads = gradients.flatten()
        # numpy.ndarray object has no attribute 'topk'
        flattened_grads = torch.tensor(flattened_grads, requires_grad=True)
        top_indices = flattened_grads.topk(top_pixels).indices
        top_indices = torch.stack([top_indices // 32, top_indices % 32])
        
        
        # Display the masked image
        image_copy = inputs.clone().detach()
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        new_image = torch.zeros((3, 32, 32))
        new_image[:, : , :] = 0.5
        new_image[:, top_indices[0], top_indices[1]] = image_copy.squeeze()[:, top_indices[0], top_indices[1]]
        axes.imshow(np.transpose(new_image.cpu(), (1, 2, 0)))
        axes.axis('off')
        plt.title(f"Top {top_pixels} Pixels for Feature {feature_idx}")
        plt.savefig(f"images/grad_maps/img_{image_index}_feature_{feature_idx}_top_{top_pixels}_pixels.png")


def evaluate_top_features(dataset, num_features, model, image_index):
    # Access the specified image directly from the dataset
    inputs, label = dataset[image_index]
    inputs = inputs.unsqueeze(0)  # Add batch dimension

    # Forward pass to get features
    features, outputs, reconstruction = model(inputs)
    features = features.squeeze(0)
    features = features.view(features.size(0), -1)
    features = features.mean(dim=1)

    # Find the top `num_features` feature indices
    top_features = torch.topk(features, num_features).indices
    print("Top feature indices:", top_features)
    return top_features


train_loader, test_loader, train_dataset, test_dataset = load_data(args)
# Assuming you have a DataLoader called 'dataloader'
model = InterpretableClassifier(num_classes=10)
# model.load_state_dict(torch.load(model_save_path_interpretable))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_interpretable(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_interpretable(model, test_loader)
selected_features = evaluate_top_features(train_dataset, 5, model, 30000)
visualize_gradients(model, train_dataset, selected_features, image_index=30000)

