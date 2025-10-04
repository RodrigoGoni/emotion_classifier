"""
Utilidades para cargar y verificar el dataset de prueba AffectNet
"""
import os
from pathlib import Path
from src.utils.constants import AFFECTNET_TO_YOUR_MODEL, EMOTION_CLASSES


def verify_test_dataset():
    """Verifica que el dataset de test AffectNet esté disponible"""
    test_path = Path("data/test_affectnet")
    
    if not test_path.exists():
        return False
    
    # Verificar que existan las carpetas de emociones
    required_folders = ['0_Anger', '1_Disgust', '2_Fear', '3_Happy', '4_Sad', '5_Surprise', '6_Neutral']
    
    for folder in required_folders:
        folder_path = test_path / folder
        if not folder_path.exists():
            return False
        
        # Verificar que tenga al menos algunas imágenes
        images = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        if len(images) == 0:
            return False
    
    return True


def load_test_affectnet_dataset():
    """Carga el dataset de test AffectNet y mapea las clases"""
    test_path = Path("data/test_affectnet")
    images_data = []
    
    # Mapeo de nombres de carpetas AffectNet a nuestras clases
    affectnet_folders = {
        '0_Anger': 0,
        '1_Disgust': 1, 
        '2_Fear': 2,
        '3_Happy': 3,
        '4_Sad': 4,
        '5_Surprise': 5,
        '6_Neutral': 6
    }
    
    for folder_name, affectnet_class in affectnet_folders.items():
        folder_path = test_path / folder_name
        
        if not folder_path.exists():
            continue
            
        # Obtener la clase correspondiente en nuestro modelo
        our_class = AFFECTNET_TO_YOUR_MODEL.get(affectnet_class)
        
        if our_class is None:
            continue  # Saltar clases que no mapeamos
        
        # Obtener todas las imágenes de la carpeta
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(folder_path.glob(extension))
        
        # Agregar información de cada imagen
        for image_path in image_files:
            image_info = {
                'path': image_path,
                'true_class': our_class,
                'true_emotion': EMOTION_CLASSES[our_class],
                'affectnet_class': affectnet_class,
                'affectnet_name': folder_name
            }
            images_data.append(image_info)
    
    return images_data