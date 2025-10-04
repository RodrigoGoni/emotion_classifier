"""
Script para crear un dataset de test fijo del AffectNet
"""
import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.constants import AFFECTNET_CLASSES, AFFECTNET_TO_YOUR_MODEL, EMOTION_CLASSES


def download_affectnet():
    """Descargar dataset AffectNet"""
    try:
        import kagglehub
    except ImportError:
        os.system("pip install kagglehub")
        import kagglehub
    
    return kagglehub.dataset_download("yakhyokhuja/affectnetaligned")


def collect_images(dataset_path, images_per_class=100):
    """Recolectar imágenes del dataset AffectNet"""
    images_by_class = defaultdict(list)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root)
        
        if folder_name.isdigit():
            affectnet_class = int(folder_name)
            
            if affectnet_class in AFFECTNET_CLASSES:
                mapped_class = AFFECTNET_TO_YOUR_MODEL.get(affectnet_class)
                
                if mapped_class is None:
                    continue
                
                for file in files:
                    if Path(file).suffix.lower() in valid_extensions:
                        image_path = Path(root) / file
                        if image_path.exists():
                            images_by_class[affectnet_class].append(image_path)
    
    # Seleccionar muestra aleatoria
    selected_images = {}
    random.seed(42)
    
    for affectnet_class, images in images_by_class.items():
        if len(images) >= images_per_class:
            selected = random.sample(images, images_per_class)
        else:
            selected = images
        
        selected_images[affectnet_class] = selected
    
    return selected_images


def create_dataset(selected_images, output_dir):
    """Crear el dataset de test"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for affectnet_class in selected_images.keys():
        affectnet_name = AFFECTNET_CLASSES[affectnet_class]
        class_dir = output_dir / f"{affectnet_class}_{affectnet_name}"
        class_dir.mkdir(exist_ok=True)
        
        for i, src_path in enumerate(selected_images[affectnet_class]):
            file_extension = src_path.suffix
            dst_filename = f"{affectnet_class}_{affectnet_name}_{i+1:03d}{file_extension}"
            dst_path = class_dir / dst_filename
            
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copiando {src_path}: {e}")
                continue
    
    return output_dir


def main():
    """Función principal"""
    print("Creando dataset de test AffectNet...")
    
    OUTPUT_DIR = "data/test_affectnet"
    
    # Verificar si ya existe
    if Path(OUTPUT_DIR).exists():
        response = input("El dataset ya existe. Recrear? (s/n): ").lower()
        if response != 's':
            return
        shutil.rmtree(OUTPUT_DIR)
    
    # Descargar dataset
    print("Descargando dataset AffectNet...")
    affectnet_path = download_affectnet()
    
    # Recolectar imágenes
    selected_images = collect_images(affectnet_path, 100)
    
    if not selected_images:
        print("No se encontraron imágenes")
        return
    
    # Crear dataset
    created_dir = create_dataset(selected_images, OUTPUT_DIR)
    
    print(f"Dataset creado en: {created_dir}")
    total = sum(len(images) for images in selected_images.values())
    print(f"Total de imágenes: {total}")


if __name__ == "__main__":
    main()