"""
Script para crear un dataset balanceado de emociones faciales
"""
import os
import sys
import shutil
import random
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.dataset import EmotionDataset


def create_balanced_dataset(source_dir, target_dir, target_size=3000):
    """Crea dataset balanceado con oversampling agresivo"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    dataset = EmotionDataset(source_dir)
    class_counts = dataset.get_class_counts()
    
    print("Distribución original:")
    for emotion, count in class_counts.items():
        print(f"  {emotion}: {count}")
    
    print(f"\nBalanceando a {target_size} muestras por clase...")
    
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
            # Copiar originales
            for img in image_files:
                shutil.copy2(img, target_emotion_dir / img.name)
            
            # Duplicar hasta target_size
            needed = target_size - current_count
            for i in range(needed):
                source_img = random.choice(image_files)
                new_name = f"{source_img.stem}_dup_{i:04d}{source_img.suffix}"
                shutil.copy2(source_img, target_emotion_dir / new_name)
        
        print(f"  {emotion}: {current_count} -> {target_size}")


def split_dataset(balanced_dir, output_dir, train_ratio=0.8):
    """Divide dataset en train/val"""
    balanced_path = Path(balanced_dir)
    output_path = Path(output_dir)
    
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion_dir in balanced_path.iterdir():
        if not emotion_dir.is_dir():
            continue
            
        emotion = emotion_dir.name
        images = list(emotion_dir.glob('*.jpg'))
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        (train_dir / emotion).mkdir(exist_ok=True)
        (val_dir / emotion).mkdir(exist_ok=True)
        
        for img in train_images:
            shutil.copy2(img, train_dir / emotion / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / emotion / img.name)
    
    print(f"Split: {len(train_images)} train, {len(val_images)} val por clase")


def main():
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / "data" / "raw" / "dataset_emociones" / "train"
    balanced_dir = base_dir / "data" / "processed" / "balanced"
    final_dir = base_dir / "data" / "processed"
    
    if not source_dir.exists():
        print(f"Error: {source_dir} no existe")
        return
    
    create_balanced_dataset(str(source_dir), str(balanced_dir), target_size=3000)
    split_dataset(str(balanced_dir), str(final_dir))
    print("Dataset balanceado creado")


if __name__ == "__main__":
    random.seed(42)
    main()