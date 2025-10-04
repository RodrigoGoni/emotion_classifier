"""
Script para crear un dataset balanceado de emociones faciales
"""
import os
import sys
import shutil
import random
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import EmotionDataset


def create_balanced_dataset(source_dir, target_dir, target_size=3000):
    """Crea dataset balanceado con oversampling agresivo"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    dataset = EmotionDataset(source_dir)
    
    for emotion in dataset.classes:
        source_emotion_dir = source_path / emotion
        target_emotion_dir = target_path / emotion
        target_emotion_dir.mkdir(exist_ok=True)
        
        image_files = list(source_emotion_dir.glob('*.jpg'))
        current_count = len(image_files)
        
        if current_count >= target_size:
            selected = random.sample(image_files, target_size)
            for img in selected:
                shutil.copy2(img, target_emotion_dir / img.name)
        else:
            for img in image_files:
                shutil.copy2(img, target_emotion_dir / img.name)
            
            needed = target_size - current_count
            for i in range(needed):
                source_img = random.choice(image_files)
                new_name = f"{source_img.stem}_dup_{i:04d}{source_img.suffix}"
                shutil.copy2(source_img, target_emotion_dir / new_name)
    
    print(f"Entrenamiento: {len(dataset.classes) * target_size} imágenes balanceadas")


def copy_validation_dataset(source_val_dir, target_val_dir):
    """Copia el dataset de validación original sin modificaciones"""
    source_path = Path(source_val_dir)
    target_path = Path(target_val_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    total_val_images = 0
    for emotion_dir in source_path.iterdir():
        if not emotion_dir.is_dir():
            continue
            
        emotion = emotion_dir.name
        target_emotion_dir = target_path / emotion
        target_emotion_dir.mkdir(exist_ok=True)
        
        images = list(emotion_dir.glob('*.jpg'))
        for img in images:
            shutil.copy2(img, target_emotion_dir / img.name)
        
        total_val_images += len(images)
    
    print(f"Validación: {total_val_images} imágenes")
    return total_val_images


def main():
    base_dir = Path(__file__).parent.parent
    source_train_dir = base_dir / "data" / "raw" / "dataset_emociones" / "train"
    source_val_dir = base_dir / "data" / "raw" / "dataset_emociones" / "validation"
    balanced_dir = base_dir / "data" / "processed" / "balanced"
    final_dir = base_dir / "data" / "processed"
    
    if not source_train_dir.exists() or not source_val_dir.exists():
        print("Error: directorios fuente no encontrados")
        return
    
    create_balanced_dataset(str(source_train_dir), str(balanced_dir), target_size=3000)
    
    train_final_dir = final_dir / "train"
    if train_final_dir.exists():
        shutil.rmtree(train_final_dir)
    shutil.copytree(balanced_dir, train_final_dir)
    
    val_final_dir = final_dir / "val"
    copy_validation_dataset(str(source_val_dir), str(val_final_dir))
    
    print("Dataset procesado completo")


if __name__ == "__main__":
    random.seed(42)
    main()