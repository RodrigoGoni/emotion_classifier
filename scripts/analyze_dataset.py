"""
Script simple para analizar el dataset de emociones
"""
import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_train_transforms


def main():
    dataset_path = "data/raw/dataset_emociones/train"
    my_transforms = get_train_transforms()
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el directorio {dataset_path}")
        return
    
    print("Loading dataset...")
    dataset = EmotionDataset(root=dataset_path, transform=my_transforms)
    
    dataset.print_info()
    dataset.plot_histogram()
    #plot some images from dartaset
    dataset.plot_sample_images(num_images=12)

if __name__ == "__main__":
    main()