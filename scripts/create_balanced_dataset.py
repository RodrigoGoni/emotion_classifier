"""
Script para crear un dataset balanceado de emociones faciales con versionado
"""
import os
import sys
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import EmotionDataset
from src.utils.config_manager import ConfigManager


def create_balanced_dataset(config_manager: ConfigManager):
    """Crea dataset balanceado usando configuración centralizada"""
    
    # Obtener configuración
    data_config = config_manager.get_config('data')
    version_info = config_manager.get_config('version_control')
    
    print(f"=== CREACIÓN DE DATASET BALANCEADO ===")
    print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
    
    # Rutas
    source_dir = data_config['raw_data_path']
    processed_path = Path(data_config['processed_path'])
    train_path = Path(data_config['train_path'])
    val_path = Path(data_config['val_path'])
    
    # Configuración de balanceado
    target_size = data_config.get('target_samples_per_class', 3000)
    validation_split = data_config.get('validation_split', 0.2)
    random_seed = data_config.get('random_seed', 42)
    
    print(f"Fuente: {source_dir}")
    print(f"Destino procesado: {processed_path}")
    print(f"Muestras por clase: {target_size}")
    print(f"Split validación: {validation_split}")
    print(f"Semilla aleatoria: {random_seed}")
    
    # Verificar fuente
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f" Error: {source_path} no existe")
        return False
    
    # Crear directorios
    processed_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Configurar semilla
    random.seed(random_seed)
    
    try:
        dataset = EmotionDataset(source_dir)
        print(f" Dataset cargado: {len(dataset.classes)} clases encontradas")
        
        total_train_images = 0
        total_val_images = 0
        
        for emotion in dataset.classes:
            print(f"\nProcesando clase: {emotion}")
            
            source_emotion_dir = source_path / emotion
            train_emotion_dir = train_path / emotion
            val_emotion_dir = val_path / emotion
            
            # Crear directorios por emoción
            train_emotion_dir.mkdir(exist_ok=True)
            val_emotion_dir.mkdir(exist_ok=True)
            
            # Obtener todas las imágenes
            image_files = list(source_emotion_dir.glob('*.jpg'))
            current_count = len(image_files)
            print(f"  Imágenes originales: {current_count}")
            
            # Generar dataset balanceado
            if current_count >= target_size:
                # Si tenemos suficientes, seleccionar aleatoriamente
                selected = random.sample(image_files, target_size)
                print(f"  Seleccionadas aleatoriamente: {len(selected)}")
            else:
                # Si no tenemos suficientes, aplicar oversampling
                selected = image_files.copy()
                needed = target_size - current_count
                
                # Duplicar imágenes para llegar al target
                for i in range(needed):
                    source_img = random.choice(image_files)
                    # Crear nombre único para duplicado
                    new_name = f"{source_img.stem}_dup_{i:04d}{source_img.suffix}"
                    selected.append((source_img, new_name))
                
                print(f"  Originales: {current_count}, Duplicadas: {needed}, Total: {len(selected)}")
            
            # Dividir en train/val
            if isinstance(selected[0], tuple):
                # Caso con duplicados
                train_set, val_set = train_test_split(
                    selected, 
                    test_size=validation_split, 
                    random_state=random_seed
                )
            else:
                # Caso normal
                train_set, val_set = train_test_split(
                    selected, 
                    test_size=validation_split, 
                    random_state=random_seed
                )
            
            # Copiar archivos de entrenamiento
            for item in train_set:
                if isinstance(item, tuple):
                    source_img, new_name = item
                    dest_path = train_emotion_dir / new_name
                else:
                    source_img = item
                    dest_path = train_emotion_dir / source_img.name
                
                shutil.copy2(source_img, dest_path)
            
            # Copiar archivos de validación
            for item in val_set:
                if isinstance(item, tuple):
                    source_img, new_name = item
                    dest_path = val_emotion_dir / new_name
                else:
                    source_img = item
                    dest_path = val_emotion_dir / source_img.name
                
                shutil.copy2(source_img, dest_path)
            
            train_count = len(train_set)
            val_count = len(val_set)
            total_train_images += train_count
            total_val_images += val_count
            
            print(f"   Train: {train_count}, Val: {val_count}")
        
        print(f"\n=== RESUMEN ===")
        print(f" Total imágenes de entrenamiento: {total_train_images}")
        print(f" Total imágenes de validación: {total_val_images}")
        print(f" Dataset balanceado creado exitosamente")
        
        # Guardar metadatos del dataset
        dataset_metadata = {
            "version": version_info['version'],
            "timestamp": config_manager.timestamp,
            "source_path": str(source_path),
            "target_samples_per_class": target_size,
            "validation_split": validation_split,
            "random_seed": random_seed,
            "classes": dataset.classes,
            "train_samples": total_train_images,
            "val_samples": total_val_images,
            "total_samples": total_train_images + total_val_images
        }
        
        metadata_file = processed_path / "dataset_metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
        
        print(f" Metadatos guardados en: {metadata_file}")
        
        return True
        
    except Exception as e:
        print(f" Error durante la creación del dataset: {e}")
        return False


def main():
    """Función principal"""
    print("=== CREADOR DE DATASET BALANCEADO ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager()
        print(" ConfigManager inicializado")
        
        # Crear dataset balanceado
        success = create_balanced_dataset(config_manager)
        
        if success:
            print("\n Proceso completado exitosamente")
        else:
            print("\n Proceso falló")
            
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    main()