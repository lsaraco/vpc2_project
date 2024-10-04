"""Clase para cargar dataset."""

from torch.utils.data import Dataset
import h5py

class MPIIFaceGaze(Dataset):

    dataset_path:str|None = None

    def __init__(self, dataset_path,imgs_per_individual):
        self.dataset_path = dataset_path
        self.imgs_per_individual = imgs_per_individual

        self.total_images = self.imgs_per_individual * 10 #Hay 10 personas en el dataset the MPIIFaceGaze

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        #TODO: Actualizar!!
        # Obtener el índice del dataset original
        original_idx = self.indices[idx]

        # Obtener la imagen y la etiqueta correspondientes
        image, label = self.original_dataset[original_idx]

        # Remapear la etiqueta
        remapped_label = self.label_map[label]

        return image, remapped_label