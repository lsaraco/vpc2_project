"""Modelos para usar con MPIIFaceGaze."""

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import numpy as np


class GazeEstimation_ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super(GazeEstimation_ResNet18, self).__init__()

        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if pretrained else models.resnet18
        
        # Modify the final fully connected layer to output 2 values (yaw, pitch)
        num_ftrs = self.resnet18.fc.in_features
        
        #self.resnet18.fc = nn.Linear(num_ftrs, 2)  # Output layer for regression (pitch, yaw)
        # Le agrego una capa fully conected mas antes de la salida de (pitch, yaw)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  
            nn.ReLU(),                 
            nn.Linear(512, 2)          
        )

        # Atributos para guardar las curvas de entrenamiento y validacion
        self.epoch_train_loss = []
        self.epoch_val_loss = []


    def forward(self, x):
        # Forward pass through the ResNet18 model
        return self.resnet18(x)
    

    def train_one_epoch(self, dataloader, criterion, optimizer, device):
        self.train()  # Set the model to training mode
      
        running_loss = []
        bar = tqdm(dataloader)

        for inputs, labels in bar:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            
            bar.set_description(f"Train loss {np.mean(running_loss):.5f}")
         
        return np.mean(running_loss)


    def fit(self, train_loader, val_loader, criterion, optimizer, epochs, device, save_model=None):
        
        self.epoch_train_loss = []
        self.epoch_val_loss = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            train_loss = self.train_one_epoch(train_loader, criterion, optimizer, device)
            
            self.epoch_train_loss.append(train_loss)

            # Validate at the end of each epoch
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion, device)
                

                # Si indio un nombre para el archivo, guardo el mejor modelo
                if save_model is not None:
                    if len(self.epoch_val_loss) == 0:
                        torch.save(self.state_dict(), save_model)
                    else:
                        if val_loss < self.epoch_val_loss[-1]:
                            torch.save(self.state_dict(), save_model)

                self.epoch_val_loss.append(val_loss)
                 
            print(f"Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            
            
    

    def evaluate(self, dataloader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = []
        bar = tqdm(dataloader)

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in bar:
                inputs, labels = inputs.float().to(device), labels.float().to(device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                running_loss.append(loss.item())

                bar.set_description(f"Validation loss {np.mean(running_loss):.5f}")
        
        return np.mean(running_loss)

