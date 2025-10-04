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


def download_affectnet_dataset(data_dir="data/external"):
    """Descarga el dataset AffectNet desde Kaggle"""
    try:
        import kagglehub
        
        print("Descargando AffectNet dataset desde Kaggle...")
        path = kagglehub.dataset_download("yakhyokhuja/affectnetaligned")
        print(f"Dataset AffectNet descargado en: {path}")
        
        # Crear enlace simbólico o copiar a data/external
        external_dir = Path(data_dir)
        external_dir.mkdir(parents=True, exist_ok=True)
        
        affectnet_link = external_dir / "affectnet"
        if not affectnet_link.exists():
            affectnet_link.symlink_to(Path(path))
            print(f"Enlace creado: {affectnet_link} -> {path}")
        
        return str(path)
        
    except ImportError:
        print("kagglehub no está instalado. Instalando...")
        os.system("pip install kagglehub")
        return download_affectnet_dataset(data_dir)
    except Exception as e:
        print(f"Error descargando AffectNet: {e}")
        return None


def verify_dataset_structure(data_dir="data/raw"):
    """Verifica la estructura del dataset"""
    # Verificar si existe la estructura dataset_emociones/train
    train_dir = os.path.join(data_dir, "dataset_emociones", "train")
    emotions = ['alegria', 'disgusto', 'enojo', 'miedo', 'seriedad', 'sorpresa', 'tristeza']
    
    if not os.path.exists(train_dir):
        print(f"Directorio {train_dir} no existe")
        return False
    
    print(f"Verificando estructura en: {train_dir}")
    for emotion in emotions:
        emotion_path = os.path.join(train_dir, emotion)
        if os.path.exists(emotion_path):
            num_images = len([f for f in os.listdir(emotion_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{emotion}: {num_images} imágenes")
        else:
            print(f"Falta carpeta: {emotion}")
            return False
    
    return True


def verify_affectnet_structure(data_dir="data/external"):
    """Verifica la estructura del dataset AffectNet"""
    affectnet_dir = Path(data_dir) / "affectnet"
    
    if not affectnet_dir.exists():
        print(f"Directorio AffectNet no existe en: {affectnet_dir}")
        return False
    
    print(f"Verificando estructura AffectNet en: {affectnet_dir}")
    
    # Listar contenido del dataset
    for root, dirs, files in os.walk(affectnet_dir):
        level = root.replace(str(affectnet_dir), '').count(os.sep)
        if level > 2:  # Limitar profundidad
            continue
            
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Mostrar algunos archivos de ejemplo
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            subindent = ' ' * 2 * (level + 1)
            for file in image_files[:3]:
                print(f'{subindent}{file}')
            if len(image_files) > 3:
                print(f'{subindent}... y {len(image_files) - 3} imágenes más')
    
    return True


def get_affectnet_sample_images(data_dir="data/external", num_samples=6):
    """Obtiene imágenes de muestra del dataset AffectNet para testing"""
    affectnet_dir = Path(data_dir) / "affectnet"
    
    if not affectnet_dir.exists():
        print("Dataset AffectNet no encontrado")
        return []
    
    sample_images = []
    
    # Buscar imágenes en el dataset
    for root, dirs, files in os.walk(affectnet_dir):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:num_samples]:
            img_path = Path(root) / img_file
            sample_images.append(str(img_path))
            
            if len(sample_images) >= num_samples:
                break
                
        if len(sample_images) >= num_samples:
            break
    
    print(f"Encontradas {len(sample_images)} imágenes de muestra de AffectNet")
    return sample_images


if __name__ == "__main__":
    print("=== Descarga de Datasets ===")
    
    # Descargar dataset principal
    print("\n1. Descargando dataset principal de emociones...")
    download_emotion_dataset()
    verify_dataset_structure()
    
    # Descargar dataset AffectNet para testing
    print("\n2. Descargando AffectNet dataset para pruebas...")
    affectnet_path = download_affectnet_dataset()
    
    if affectnet_path:
        verify_affectnet_structure()
        
        # Obtener imágenes de muestra
        sample_images = get_affectnet_sample_images()
        if sample_images:
            print(f"\nImágenes de muestra para testing:")
            for i, img_path in enumerate(sample_images, 1):
                print(f"  {i}. {img_path}")
    
    print("\n=== Descarga completada ===")