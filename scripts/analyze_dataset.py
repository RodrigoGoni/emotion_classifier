"""Script para analizar el dataset de emociones con gestión de versiones"""
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_train_transforms
from src.utils.config_manager import ConfigManager


def analyze_dataset_with_config(config_manager: ConfigManager):
    """Analiza el dataset usando configuración centralizada"""
    data_config = config_manager.get_config('data')
    version_info = config_manager.get_config('version_control')
    
    print(f"=== ANÁLISIS DE DATASET ===")
    print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
    
    # Usar rutas de configuración
    dataset_path = Path(data_config['raw_data_path']) / "train"
    
    if not dataset_path.exists():
        print(f"Error: No se encontró el directorio {dataset_path}")
        return False
    
    print(f"Analizando dataset: {dataset_path}")
    
    # Cargar dataset con transformaciones de configuración
    transforms = get_train_transforms()
    dataset = EmotionDataset(root=str(dataset_path), transform=transforms)
    
    # Crear directorio de resultados versionado
    results_dir = config_manager.get_results_dir() / "dataset_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Guardando análisis en: {results_dir}")
    
    # Análisis del dataset
    print("Dataset Information:")
    print(f"Total images: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    print()
    
    # Información de clases
    class_counts = dataset.get_class_counts()
    print("Images per class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Información de dimensiones
    dims = dataset.get_image_dimensions()
    print(f"\nImage dimensions (average):")
    print(f"  Width: {dims['avg_width']}px")
    print(f"  Height: {dims['avg_height']}px")
    print(f"  Samples analyzed: {dims['samples_analyzed']}")
    print()
    
    # Generar visualizaciones en directorio versionado
    print("Generando visualizaciones...")
    dataset.plot_histogram(save_path=str(results_dir))
    dataset.plot_sample_images(num_images=12, save_path=str(results_dir))
    
    histogram_path = results_dir / "class_distribution.png"
    samples_path = results_dir / "sample_images.png"
    print(f"Histograma guardado: {histogram_path}")
    print(f"Imágenes de muestra guardadas: {samples_path}")
    
    # Guardar metadatos del análisis
    analysis_metadata = {
        "timestamp": config_manager.timestamp,
        "dataset_path": str(dataset_path),
        "total_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "classes": dataset.classes,
        "samples_per_class": class_counts,
        "image_dimensions": dims,
        "version_info": version_info
    }
    
    metadata_file = results_dir / "analysis_metadata.json"
    import json
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Metadatos guardados: {metadata_file}")
    
    return True


def main():
    """Función principal"""
    print("=== ANALIZADOR DE DATASET ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager()
        print(" ConfigManager inicializado")
        
        # Analizar dataset
        success = analyze_dataset_with_config(config_manager)
        
        if success:
            print("\n Análisis completado exitosamente")
        else:
            print("\n Análisis falló")
            
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    main()