"""Archivo de funciones auxiliares como plot de imagenes, annotations, etc."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import logging

def rgb(img):
    """Convierte imagenes de BGR a RGB, para poder ser mostradas en un notebook."""
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Function to reverse the normalization
def inverse_normalize(tensor, mean, std):
    mean = mean[:, None, None]  # Reshape for broadcasting
    std = std[:, None, None]
    tensor = tensor * std + mean  # Denormalize
    return tensor

def draw_gaze360_arrow(img, eye_gaze_2d, eye_position, scale=10,color=(0,255,0)):
    """
    Draw an arrow on the image pointing in the direction of eye gaze.

    Parameters:
    - img: The image (RGB format) on which to draw.
    - eye_gaze_2d: A 2D tensor containing pitch (vertical angle) and yaw (horizontal angle).
    - eye_position: A tuple (x, y) representing the position of the eye in the image (where the arrow starts).
    - scale: A scaling factor to control the length of the arrow.
    """
    # Extract yaw and pitch from the 2D tensor
    yaw = eye_gaze_2d[0].item()  # Horizontal angle (theta)
    pitch = eye_gaze_2d[1].item()  # Vertical angle (phi)

    # Project the 2D gaze direction to x and y components on the image
    x_2d = math.cos(pitch) * math.sin(yaw)  # Horizontal direction
    y_2d = math.sin(pitch)  # Vertical direction

    if eye_position=="center":
        height,width,_ = img.shape
        eye_position = [int(width/2),int(height/2)]


    # Compute the end point of the arrow, scaled by the 'scale' factor
    end_point = (
        int(eye_position[0] - x_2d * scale),
        int(eye_position[1] - y_2d * scale)  # Negative because the y-axis is inverted in image coordinates
    )
    
    # Draw the arrow on the image
    img_annotated = cv2.arrowedLine(img.copy(), eye_position, end_point, color, 1, tipLength=0.3)

    return img_annotated

def put_gaze_annotation(img,gaze,method="ALL",color=None,label=None,label_y=10):
    """Agrega una flecha sobre la imagen apuntando en la dirección de la mirada."""

    h, w, _ = img.shape

    # Obtengo centro de la cara
    face_center = (int(w / 2), int(h *0.6/ 2))

    # Creo copia de la imagen para que los annotations solo estén en la copia
    img2 = img.copy()

    # Parámetros - configuraciones
    color_green = (0, 255, 0)
    color_blue = (0, 0, 255) 
    color_red = (255, 0, 0) 
    thickness = 2

    # Calculo vector de la flecha - Método 1
    if method==1 or method=="ALL":
        arrow_length = 0.7*np.sqrt(h**2+w**2)
        pitch = gaze[0]
        yaw = gaze[1]
        dx = np.cos(pitch)*np.sin(yaw)
        dy = np.sin(pitch)
        end_point = (
            int(face_center[0] - arrow_length * dx),
            int(face_center[1] - arrow_length * dy) 
        )
        color_ = color_green if color is None else color
        cv2.arrowedLine(img2, face_center, end_point, color_, thickness, tipLength=0.2)

    # Calculo vector de la flecha convirtiendo primero los ángulos normalizados a radianes
    # y con dx solo dependiente de yaw
    if method==2 or method=="ALL":
        arrow_length = 0.3*np.sqrt(h**2+w**2)
        pitch = gaze[0]*np.pi
        yaw = gaze[1]*np.pi
        dx = np.sin(yaw)
        dy = np.sin(pitch)    
        end_point = (
            int(face_center[0] - arrow_length * dx),  # Horizontal component (yaw)
            int(face_center[1] - arrow_length * dy)   # Vertical component (pitch, inverted Y)
        )
        color_ = color_green if color is None else color
        cv2.arrowedLine(img2, face_center, end_point, color_, thickness, tipLength=0.2)


    # Idem método 1 pero convirtiendo el ángulo a radianes previamente
    if method==3 or method=="ALL":
        arrow_length = 0.3*np.sqrt(h**2+w**2)
        pitch = gaze[0]*np.pi
        yaw = gaze[1]*np.pi
        dx = np.cos(pitch)*np.sin(yaw)
        dy = np.sin(pitch)
        end_point = (
            int(face_center[0] - arrow_length * dx),  # Horizontal component (yaw)
            int(face_center[1] - arrow_length * dy)   # Vertical component (pitch, inverted Y)
        )
        color_ = color_green if color is None else color
        cv2.arrowedLine(img2, face_center, end_point, color_, thickness, tipLength=0.2)

    # Se agrega label para identificar las anotaciones
    if label:
        dash_start = (5, label_y)
        dash_end = (15, label_y)
        cv2.line(img2, dash_start, dash_end, color, 2)
        text_position = (dash_end[0] + 5, dash_end[1] + 2)
        cv2.putText(img2, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, color, 1 )



    # Return de la imagen con las anotaciones
    return img2

def angle_to_2d(pitch_yaw_arraw):
    """Convierte el ángulo tridimensional (yaw,pitch) a 2D, mediante una proyección."""
    yaw = pitch_yaw_arraw[:, 0]
    pitch = pitch_yaw_arraw[:, 1]

    x = np.cos(pitch)*np.sin(yaw)
    y = np.sin(pitch)
    angles_2d = np.rad2deg(np.arctan2(y, x)) + 90

    angles_2d[angles_2d < 0] += 360
    angles_2d[angles_2d > 360] -= 360

    return angles_2d


def is_accurate(actual_angle,predicted_angle,tolerance):
    """Función auxiliar para medir accuracy en base a tolerance grados de tolerancia."""
    res = actual_angle - predicted_angle
    mask = np.abs(res)<tolerance
    res[mask] = 1
    res[~mask] = 0
    return res

def plot_metrics(model,title=None):
    """Funcion para plotear rápidamente las métricas de cada modelo."""
    # Grafico las curvas de entrenamiento y validacion
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Si hay titulo lo agrego arriba de todo
    if title and type(title) is str:
        fig.suptitle(title,fontsize=14)

    # Loss subplot
    ax1.set_title("Epoch Loss")
    ax1.plot(range(len(model.epoch_train_loss)), model.epoch_train_loss, label='Train Loss')
    ax1.plot(range(len(model.epoch_val_loss)), model.epoch_val_loss, label='Validation Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()  

    # Zoom en curvas de loss, se evitan los primeros 'start_val' valores 
    # que tienen loss muy alto y no dejan ver bien el resto
    start_val = 2
    if len(model.epoch_val_loss) > start_val:
        ax1.set_xlim([start_val,len(model.epoch_val_loss)])
        ax1.set_ylim([min(model.epoch_val_loss[start_val:]+model.epoch_train_loss[start_val:])*0.9,
                    max(model.epoch_val_loss[start_val:]+model.epoch_train_loss[start_val:])*1.1])

    # Accuracy subplot
    ax2.set_title("Epoch Accuracy")
    ax2.plot(range(len(model.epoch_train_accuracy)), model.epoch_train_accuracy, label='Train Accuracy')
    ax2.plot(range(len(model.epoch_val_accuracy)), model.epoch_val_accuracy, label='Validation Accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy [%]")
    ax2.grid(True)
    ax2.legend()

    # Muestro los plots
    plt.tight_layout()
    plt.show()


# Obtengo combinaciones aleatorias de los valores definidos en el diccionario de parametros
def getRandomHyperparam(n, parameters, seed=None):
    
    if seed is not None:
        random.seed(seed)
    
    keys = list(parameters.keys())

    combinaciones = []
    for i in range(n):
        
        combinacion = {key: random.choice(parameters[key]) for key in keys}
        combinaciones.append(combinacion)
    
    return combinaciones


# Set environment variable to suppress logs when using mediapipe
os.environ["GLOG_minloglevel"] = "2"  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlow logging level (if used)
os.environ["LIBGL_DEBUG"] = "quiet"
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import mediapipe as mp

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_bounding_box_from_landmarks(landmarks, image_width, image_height):
    # Extract X and Y coordinates of landmarks
    x_coords = [int(landmark.x * image_width) for landmark in landmarks]
    y_coords = [int(landmark.y * image_height) for landmark in landmarks]

    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, y_min, x_max, y_max

def detect_face_bounding_box_from_array(image):
    """
    Detects the bounding box of the head from an input image as a NumPy array.
    Args:
        image: Input image as a NumPy array (H x W x 3)
    """


    mp_face_mesh = mp.solutions.face_mesh

    image_height, image_width, _ = image.shape


    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        # Process the image
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            print("No face detected.")
            return None

        for face_landmarks in results.multi_face_landmarks:
            # Calculate bounding box
            bbox = get_bounding_box_from_landmarks(face_landmarks.landmark, image_width, image_height)
            x_min, y_min, x_max, y_max = bbox

            # Ensure bounding box is within image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width, x_max)
            y_max = min(image_height, y_max)

            # Crop the face region
            face_image = image[y_min:y_max, x_min:x_max]

            return face_image
        
def detect_eyes_region_from_array(image):
    """
    Detects the bounding box of the eyes region (with a +-5px height margin) from an input image as a NumPy array.
    Args:
        image: Input image as a NumPy array (H x W x 3)

    Returns:
        The cropped eyes region as a NumPy array or None if no face is detected.
    """
    mp_face_mesh = mp.solutions.face_mesh

    # Get the dimensions of the input image
    image_height, image_width, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        # Process the image
        results = face_mesh.process(image)

        if not results.multi_face_landmarks:
            print("No face detected.")
            return None

        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of eye landmarks
            left_eye_indices = [33, 160, 159, 158, 157, 173]  # Example indices for the left eye
            right_eye_indices = [362, 385, 386, 387, 388, 466]  # Example indices for the right eye

            # Extract landmark coordinates and scale to image dimensions
            left_eye_points = [(int(landmark.x * image_width), int(landmark.y * image_height)) for i, landmark in enumerate(face_landmarks.landmark) if i in left_eye_indices]
            right_eye_points = [(int(landmark.x * image_width), int(landmark.y * image_height)) for i, landmark in enumerate(face_landmarks.landmark) if i in right_eye_indices]

            # Combine all eye points to create a bounding box
            all_eye_points = np.array(left_eye_points + right_eye_points)
            x_min = np.min(all_eye_points[:, 0])
            y_min = np.min(all_eye_points[:, 1])
            x_max = np.max(all_eye_points[:, 0])
            y_max = np.max(all_eye_points[:, 1])

            # Add a +-5px margin to the height and width
            y_min = max(0, y_min - 5)
            y_max = min(image_height, y_max + 5)
            x_min = max(0, x_min - 5)
            x_max = min(image_width, x_max + 5)

            # Crop the eyes region
            eyes_region = image[y_min:y_max, x_min:x_max]

            return eyes_region

    return None
