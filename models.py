"""Modelos para usar con MPIIFaceGaze."""

import torch
import torch.nn as nn
from torchvision import models


class GazeEstimation_ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super(GazeEstimation_ResNet18, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=pretrained)
        
        # Modify the final fully connected layer to output 2 values (yaw, pitch)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)  # Output layer for regression (yaw, pitch)

    def forward(self, x):
        # Forward pass through the ResNet18 model
        return self.resnet18(x)
    

    def train_one_epoch(self, dataloader, criterion, optimizer, device):
        self.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Training Loss: {epoch_loss:.4f}")
        return epoch_loss

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs, device):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            train_loss = self.train_one_epoch(train_loader, criterion, optimizer, device)
            
            # Validate at the end of each epoch
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion, device)
                print(f"Validation Loss: {val_loss:.4f}")
    
    def evaluate(self, dataloader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
        
        return running_loss / len(dataloader)



