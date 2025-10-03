"""
Script para descargar el dataset de emociones
"""
import os
import zipfile
from pathlib import Path
import gdown

def download_emotion_dataset(data_dir="data/raw"):
    """Descarga el dataset de emociones desde Google Drive"""
    url = "https://drive.google.com/file/d/1auZ64-CEfa4tx16cVq9TdibsdKwQY9jN/view?usp=sharing"
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    zip_path = os.path.join(data_dir, "emotion_dataset.zip")
    
    try:
        print("Descargando dataset...")
        gdown.download(download_url, zip_path, quiet=True)
        
        print("Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        print(f"Dataset guardado en: {data_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Descarga manual desde: https://drive.google.com/file/d/1auZ64-CEfa4tx16cVq9TdibsdKwQY9jN/view?usp=sharing")

def verify_dataset_structure(data_dir="data/raw"):
    """Verifica la estructura del dataset"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    if not os.path.exists(data_dir):
        print(f"Directorio {data_dir} no existe")
        return False
    
    for emotion in emotions:
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_path):
            num_images = len([f for f in os.listdir(emotion_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{emotion}: {num_images} imágenes")
        else:
            print(f"Falta carpeta: {emotion}")
            return False
    
    return True

if __name__ == "__main__":
    download_emotion_dataset()
    verify_dataset_structure()