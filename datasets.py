"""Clase para cargar dataset."""

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from aux_functions import detect_face_bounding_box_from_array,detect_eyes_region_from_array
import torch
import numpy as np
import h5py
import math
import scipy.io




class Gaze360Dataset(Dataset):
    """Class to easily load Gaze360 dataset."""

    dataset_path:str|None = None
    images_list = None
    gaze_list = None

    def __init__(self, dataset_path,images_limit=1000,random=True,transform=None,yaw_limits=[-1.5,1.5],face_only=True):

        # Initialize variables
        self.images_list = []
        self.gaze_list = []
        self.transform = transform
        self.abs_path = str(Path(dataset_path).absolute())

        # Read dataset and get data
        data = scipy.io.loadmat(dataset_path+"/metadata.mat")
        recording = data['recording'][0]

        # Define how to pick images
        if random:
            indices = torch.randperm(len(recording)-1)
        else:
            # Define max qty of images
            # max_i = len(recording)-1 if images_limit is None else images_limit  
            # indices = range(0,max_i)  
            indices = range(0,len(recording)-1)
        
        # Load all images to images_list and gaze_list
        cnt = 0
        for i in indices:

            # Limit counter
            if cnt >= images_limit:
                break

            # Get image
            img_path = self.load_gaze360_image(self.abs_path,i,data)
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image.copy(), dtype=np.uint8)

            # Get labels
            eye_gaze_2d = self.get_pitch_yaw_from_dataset(data,i)
            eye_gaze_2d = np.array(eye_gaze_2d,dtype=float)

            # Filter yaw to avoid images looking back, where head is not visible
            if self.is_angle_within_limits(eye_gaze_2d,yaw_limits[0],yaw_limits[1],"yaw"):
                # If face_only is True, get only face part of the image
                if face_only:
                    # img = detect_face_bounding_box_from_array(image_np)
                    img = detect_eyes_region_from_array(image_np)
                    if img is None: #Discard images where face was not detected
                        continue
                else:
                    img = image_np
                # Append to list
                self.images_list.append(img)
                self.gaze_list.append(eye_gaze_2d)
                cnt += 1
        
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        
        image = self.images_list[idx]
        label = self.gaze_list[idx]

        if self.transform:
            image = self.transform(image)  

        return image, label
    
    def is_angle_within_limits(self,angle_array,min,max,source):
        """Check that angle is within range."""
        if source=="pitch":
            i = 1
        else:
            i = 0
        angle = angle_array[i]
        if angle>=min and angle <=max:
            return True
        return False
    
    def load_gaze360_image(self,dataset_dir,i,data,cropType="head"):
        """Loads an image from 360 dataset."""
        recordings = data['recordings'][0]
        recording = data['recording'][0]
        frame = data['frame'][0]
        person_identity = data['person_identity'][0]

        path = dataset_dir+"/imgs/"
        path +=    str(recordings[recording[i]][0])
        path +=     "/"
        path +=     cropType
        path +=     "/"
        path +=     '%06d' % person_identity[i]
        path +=     "/"
        path +=     '%06d.jpg' % frame[i]

        return path
    
    def compute_spherical_vector(self,normalized_gaze):
        """Compute spherical vector from 3D coordinate system."""
        spherical_vector = torch.FloatTensor(2)
        # theta: horizontal angle (yaw)
        spherical_vector[0] = math.atan2(normalized_gaze[0], -normalized_gaze[2])
        # phi: vertical angle (pitch)
        spherical_vector[1] = math.asin(normalized_gaze[1])
        return spherical_vector
    
    def get_pitch_yaw_from_dataset(self,data,i):
        """Gets 3D eye gaze coordinates from dataset and converts to pitch and yaw angles."""
        gaze_dir = data['gaze_dir']
        gaze_3d = gaze_dir[i]
        

        # 2D transformations:
        eye_gaze_pitch_yaw = self.compute_spherical_vector(gaze_3d)

        return eye_gaze_pitch_yaw 

    


class MPIIFaceGaze(Dataset):

    dataset_path:str|None = None
    images_list = None
    gaze_list = None

    def __init__(self, dataset_path, transform=None, imgs_per_individual=None):
        
        self.dataset_path = dataset_path
        self.imgs_per_individual = imgs_per_individual
        #self.total_images = self.imgs_per_individual * 15 #Hay 15 personas en el dataset the MPIIFaceGaze

        self.transform = transform

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
                if imgs_per_individual is None:
                    random_indices = torch.randperm(len(images)) 
                else:
                    random_indices = torch.randperm(len(images))[:imgs_per_individual]

                for i in random_indices:
                    # Las imagenes se identifican con 4 números. Tienen un indice en el formato "xxxx", por ejemplo "0230"
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

        # Aplico transformaciones a las imagens 
        if self.transform:
            image = self.transform(image)  

        return image, label

