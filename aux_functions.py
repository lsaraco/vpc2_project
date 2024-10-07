"""Archivo de funciones auxiliares como plot de imagenes, annotations, etc."""
import cv2
import numpy as np


def rgb(img):
    """Convierte imagenes de BGR a RGB, para poder ser mostradas en un notebook."""
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def put_gaze_annotation(img,gaze,method="ALL",color=None):
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