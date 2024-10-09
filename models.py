"""Modelos para usar con MPIIFaceGaze."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from datetime import datetime
from aux_functions import angle_to_2d,is_accurate



class BaseGazeEstimationModel(nn.Module):
    """Clase base para los modelos de estimación de mirada."""
    ACCURACY_TOLERANCE = 15
    VAL_LOSS_TOLERANCE = 6 #Tolera hasta 6 epochs sin que el loss haya disminuido
    dynamic_lr = False
    lr_epochs_adjustment = 20
    lr_adjustment_ratio = 0.1
    minimum_lr = 1e-6
    

    def __init__(self,name="Base"):
        super().__init__()
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.timed_name = f"{name}_{self.get_current_time_str()}"
        self.writer =  SummaryWriter(log_dir=f"tensorboard/{self.timed_name}")
        self.model_weights_path = f"./modelos/{name}.pth"
        self.minimum_val_loss = 1e10 #Grande a propósito, para que el primer valor de loss sea el minimo
        self.not_minimum_count = 0
        self.early_stop = False


    def get_current_time_str(self):
        """Devuelve string de tiempo en formato YYYYMMDD_hhmm"""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M")

        return timestamp_str
    
    def train_one_epoch(self, dataloader, criterion, optimizer, device):
        self.train()  # Set the model to training mode
      
        running_loss = []
        running_accuracy = []
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
            running_accuracy.append(self.calculate_accuracy(labels,outputs))
            
            bar.set_description(f"Train loss {np.mean(running_loss):.5f}")
         
        return np.mean(running_loss),np.mean(running_accuracy)


    def fit(self, train_loader, val_loader, criterion, optimizer, epochs, device, save_model=False):
        
        self.epoch_train_loss = []
        self.epoch_train_accuracy = []
        self.epoch_val_loss = []
        self.epoch_val_accuracy = []

        # Scheduler: Si el modelo empieza a dejar de mejorar (loss no desciende), empieza a reducir el learning rate.
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            # Opción 2: Cambio del learning rate manualmente luego de self.lr_epochs_adjustment epochs
            if self.dynamic_lr:
                if (epoch+1)%self.lr_epochs_adjustment == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr']*self.lr_adjustment_ratio
                        current_lr = param_group['lr']
                        if current_lr < self.minimum_lr:
                            param_group['lr'] = self.minimum_lr
                            current_lr = self.minimum_lr
                            self.dynamic_lr = False
                
            # Train por cada epoch
            train_loss,train_accuracy = self.train_one_epoch(train_loader, criterion, optimizer, device)
            
            self.epoch_train_loss.append(train_loss)
            self.epoch_train_accuracy.append(train_accuracy)
            

            # Validación por cada epoch
            if val_loader is not None:
                val_loss,val_accuracy = self.evaluate(val_loader, criterion, device)    
                            

                # Si indico un nombre para el archivo, guardo el mejor modelo
                if save_model:
                    if len(self.epoch_val_loss) == 0:
                        torch.save(self.state_dict(), self.model_weights_path)
                    else:
                        if val_loss < self.minimum_val_loss:
                            self.minimum_val_loss = val_loss
                            self.not_minimum_count = 0
                            torch.save(self.state_dict(), self.model_weights_path)
                            print(f"Nuevo mínimo de loss encontrado! Guardando modelo en {self.model_weights_path}")
                        else:
                            self.not_minimum_count += 1
                            if self.early_stop and self.not_minimum_count >= self.VAL_LOSS_TOLERANCE:
                                print(f"Se interrumpió el entrenamiento porque en los últimos {self.VAL_LOSS_TOLERANCE} epochs no se logró menor loss que el mínimo (min:{self.minimum_val_loss})")
                                break


                self.epoch_val_loss.append(val_loss)
                self.epoch_val_accuracy.append(val_accuracy)

                #Lr scheduler
                # scheduler.step(val_loss) # Necesario para el scheduler que reduce lr
                # current_lr = scheduler.get_last_lr()[0]
            if self.writer:
                self.writer.add_scalar("Loss/train",train_loss,epoch)
                self.writer.add_scalar("Loss/test",val_loss,epoch)
                self.writer.add_scalar("Accuracy/train",train_accuracy,epoch)
                self.writer.add_scalar("Accuracy/test",val_accuracy,epoch)
            print(f"[Lr:{current_lr}] Training Loss / Accuracy: {train_loss:.4f} / {train_accuracy*100:.2f}% | Validation Loss / Accuracy: {val_loss:.4f} / {val_accuracy*100:.2f}%")
        
        if self.writer:
            self.writer.flush()
            
    

    def evaluate(self, dataloader, criterion, device):
        self.eval()  # Set the model to evaluation mode
        running_loss = []
        running_accuracy = []
        bar = tqdm(dataloader)

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in bar:
                inputs, labels = inputs.float().to(device), labels.float().to(device)
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                running_loss.append(loss.item())                
                running_accuracy.append(self.calculate_accuracy(labels,outputs))

                bar.set_description(f"Validation loss {np.mean(running_loss):.5f}")
        
        return np.mean(running_loss),np.mean(running_accuracy)
    

    def calculate_accuracy(self,labels,outputs):
        """Función auxiliar para calcular accuracy según label y outputs."""
        angle_actual = angle_to_2d(labels.detach().cpu().numpy())
        angle_predicted = angle_to_2d(outputs.detach().cpu().numpy())
        accuracy_array = is_accurate(angle_actual,angle_predicted,self.ACCURACY_TOLERANCE)
        return np.mean(accuracy_array)
    
    def close_writer(self):
        """Cierra el writer de tensorboard."""
        if self.writer:
            self.writer.close()



class GazeEstimation_ResNet18(BaseGazeEstimationModel):

    def __init__(self,name="GazeEstimation_ResNet18", pretrained=True, debug=False):
        super().__init__(name=name)

        # Configuración de parámetros respecto al modelo base
        self.dynamic_lr = True

        # Partimos de una resnet18 pre entrenada
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) if pretrained else models.resnet18

        # Se congelan todos los parámetros de las primeras capas ya que a pesar de la diferencia entre
        # ImageNet y este dataset se supone que las primeras capas capturan rasgos generales como bordes
        # Capas a congelar: initial layer y layer1
        for name, param in self.resnet18.named_parameters():
             if "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Se modificará la última capa fully connected
        num_ftrs = self.resnet18.fc.in_features
        
        # Se agregan dos capas fc más antes de la salida de (pitch, yaw)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),             
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64,2)     
        )

        # Y se la hace entrenable
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

        # Verificar qué capas quedaron entrenables y cuáles no
        if debug:
            for name, param in self.named_parameters():
                print(f"{name}: {'Entrenable' if param.requires_grad else 'No entrenable'}")
            


    def forward(self, x):
        return self.resnet18(x)
    

class GazeEstimation_ResNet34(BaseGazeEstimationModel):

    def __init__(self,name="GazeEstimation_ResNet34", pretrained=True, debug=False):
        super().__init__(name=name)

        # Configuración de parámetros respecto al modelo base
        self.dynamic_lr = True

        # Partimos de una resnet34 pre entrenada
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT) if pretrained else models.resnet34
        
        # Se congelan todos los parámetros de las primeras capas ya que a pesar de la diferencia entre
        # ImageNet y este dataset se supone que las primeras capas capturan rasgos generales como bordes
        # Capas a congelar: initial layer y layer1
        for name, param in self.resnet34.named_parameters():
            if "layer1" not in name and "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Se modificará la última capa fully connected
        num_ftrs = self.resnet34.fc.in_features


        # Se agregan dos capas fc más antes de la salida de (pitch, yaw)
        self.resnet34.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),             
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Linear(64,2)     
        )

        # Y se la hace entrenable
        for param in self.resnet34.fc.parameters():
            param.requires_grad = True

        # Verificar qué capas quedaron entrenables y cuáles no
        if debug:
            for name, param in self.named_parameters():
                print(f"{name}: {'Entrenable' if param.requires_grad else 'No entrenable'}")

    def forward(self, x):
        return self.resnet34(x)
    

class CNN_custom(BaseGazeEstimationModel):
  
  def conv_block(self,c_in, c_out, k=3, p='same', s=1, pk=3, ps=2, pp=1):
    return torch.nn.Sequential(                               # el módulo Sequential se encarga de hacer el forward de todo lo que tiene dentro.
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s), # conv
        torch.nn.BatchNorm2d(c_out),
        torch.nn.ReLU(),                                      # activation
        torch.nn.MaxPool2d(pk, stride=ps, padding=pp)         # pooling
    )
  
  def __init__(self,name="CNN_propia"):
    super().__init__(name=name)

    # Configuraciones
    kernel_size = 3
    in_channels = 3
    img_size = 224 # Para que sea compatible con las resnet

    # Capas de convolución
    self.conv1 = self.conv_block(in_channels, 16, k=5) #Primer capa con kernel fijo para capturar detalles de la imagen
    self.conv1_out = None
    self.conv2 = self.conv_block(16, 32, k=kernel_size)
    self.conv2_out = None
    self.drop_cnn = torch.nn.Dropout2d(p=0.3, inplace=False)
    self.conv3 = self.conv_block(32, 64, k=kernel_size)
    self.conv3_out = None
    self.conv4 = self.conv_block(64, 128, k=kernel_size)
    self.conv4_out = None
    # Parámetros para calcular automaticamente dimensones de input de la fully connected.
    # Por cómo está hecho conv_block, se conservan las dimensiones de la imágen luego de la capa de convolución pero
    # debido a la capa de max pooling la salida tiene ancho y altura reducidos a la mitad.
    qty_of_conv_blocks = 4
    final_cnn_out_channels = 128
    dim_fc_1 = int((img_size/(2**qty_of_conv_blocks))*(img_size/(2**qty_of_conv_blocks))*final_cnn_out_channels)
    # Capas fully-connected
    self.fc1 = torch.nn.Linear(dim_fc_1, 32)
    self.drop_fc = torch.nn.Dropout(p=0.4, inplace=False)
    self.fc2 = torch.nn.Linear(32, 2)

  def forward(self, x):
    self.conv1_out = self.conv1(x)
    self.conv2_out = self.drop_cnn(self.conv2(self.conv1_out))
    self.conv3_out = self.drop_cnn(self.conv3(self.conv2_out))
    self.conv4_out = self.conv4(self.conv3_out)
    y = self.conv4_out.view(self.conv4_out.shape[0], -1)
    y = self.fc1(y)
    y= self.drop_fc(y)
    y = self.fc2(y)
    return y

