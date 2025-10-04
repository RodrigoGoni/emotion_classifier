"""Script para crear un dataset balanceado de emociones faciales"""
import os
import sys
import shutil
import random
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import EmotionDataset
from src.utils.config_manager import ConfigManager


def _generate_augmented_images(source_images, output_dir, num_needed):
    """Genera imágenes aumentadas usando transformaciones reales"""
    if not source_images:
        return 0
    
    augment_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    for i in range(num_needed):
        try:
            source_img_path = random.choice(source_images)
            img = Image.open(source_img_path).convert('RGB')
            augmented_img = augment_transforms(img)
            
            aug_filename = f"{source_img_path.stem}_aug_{i:04d}.jpg"
            augmented_img.save(output_dir / aug_filename, quality=95)
        except:
            continue
    
    return num_needed


def create_balanced_dataset(config_manager: ConfigManager):
    """Crea dataset balanceado usando configuración centralizada"""
    data_config = config_manager.get_config('data')
    version_info = config_manager.get_config('version_control')
    
    print(f"Creando dataset balanceado - {version_info['project_name']} v{version_info['version']}")
    
    # Rutas
    source_dir = Path(data_config['raw_data_path']) / "train"
    processed_path = Path(data_config['processed_path'])
    train_path = Path(data_config['train_path'])
    val_path = Path(data_config['val_path'])
    
    target_size = data_config.get('target_samples_per_class', 3000)
    random_seed = data_config.get('random_seed', 42)
    random.seed(random_seed)
    
    # Crear directorios
    processed_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = EmotionDataset(str(source_dir))
        print(f"Dataset: {len(dataset.classes)} clases")
        
        total_train = 0
        total_val = 0
        
        for emotion in dataset.classes:
            print(f"Procesando {emotion}...")
            
            source_emotion_dir = source_dir / emotion
            train_emotion_dir = train_path / emotion
            val_emotion_dir = val_path / emotion
            
            train_emotion_dir.mkdir(exist_ok=True)
            val_emotion_dir.mkdir(exist_ok=True)
            
            # Obtener imágenes
            image_files = list(source_emotion_dir.glob('*.jpg'))
            current_count = len(image_files)
            
            if current_count == 0:
                continue
            
            # Generar dataset de entrenamiento balanceado
            if current_count >= target_size:
                selected = random.sample(image_files, target_size)
                for img_file in selected:
                    shutil.copy2(img_file, train_emotion_dir / img_file.name)
            else:
                # Copiar originales
                for img_file in image_files:
                    shutil.copy2(img_file, train_emotion_dir / img_file.name)
                
                # Generar augmentadas
                needed = target_size - current_count
                _generate_augmented_images(image_files, train_emotion_dir, needed)
            
            # Copiar validación
            validation_dir = Path(data_config['raw_data_path']) / "validation" / emotion
            val_count = 0
            if validation_dir.exists():
                val_images = list(validation_dir.glob('*.jpg'))
                for img_file in val_images:
                    shutil.copy2(img_file, val_emotion_dir / img_file.name)
                val_count = len(val_images)
                total_val += val_count
            
            total_train += target_size
            print(f"  Train: {target_size}, Val: {val_count}")
        
        # Guardar metadatos
        metadata = {
            "version": version_info['version'],
            "timestamp": config_manager.timestamp,
            "target_samples_per_class": target_size,
            "classes": dataset.classes,
            "train_samples": total_train,
            "val_samples": total_val,
        }
        
        with open(processed_path / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Completado: {total_train} train, {total_val} val")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Función principal"""
    try:
        config_manager = ConfigManager()
        success = create_balanced_dataset(config_manager)
        
        if success:
            print("Proceso completado")
        else:
            print("Proceso falló")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
