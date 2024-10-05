"""Clase para cargar dataset."""

from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np
import h5py

class MPIIFaceGaze(Dataset):

    dataset_path:str|None = None
    images_list = None
    gaze_list = None

    def __init__(self, dataset_path,imgs_per_individual):
        self.dataset_path = dataset_path
        self.imgs_per_individual = imgs_per_individual
        self.total_images = self.imgs_per_individual * 15 #Hay 15 personas en el dataset the MPIIFaceGaze

        # Open the file in read mode
        images_list = []
        gaze_list = []
        abs_path = Path(dataset_path).absolute()
        with h5py.File(dataset_path, 'r') as file:
            for person in file:
                
                individual_dataset = file[person]
                images = individual_dataset["image"]
                gazes = individual_dataset["gaze"]
                # pose = individual_dataset["pose"]

                # Se eligen imgs_per_individual indices random del dataset
                random_indices = torch.randperm(len(images))[:imgs_per_individual]

                for i in random_indices:
                    # Las imagenes tienen un indice en el formato "xxxx", por ejemplo "0230"
                    i_key = f"{i:04d}"
                    img = np.array(images[i_key],dtype=np.uint8)
                    gaze = np.array(gazes[i_key],dtype=float)
                    # pose = np.array(pose[i_key],dtype=float)
                    images_list.append(img)
                    gaze_list.append(gaze)
            self.images_list = images_list
            self.gaze_list = gaze_list
        if self.images_list is None or len(self.images_list) < 1:
            raise ValueError(f"Hubo un error al cargar el dataset {abs_path}")
        else:
            print(f"Dataset cargado correctamente de {abs_path}")


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        
        image = self.images_list[idx]
        label = self.gaze_list[idx]

        return image, label