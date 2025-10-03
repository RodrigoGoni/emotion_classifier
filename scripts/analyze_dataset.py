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


def main():
    dataset_path = "data/raw/dataset_emociones/train"
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el directorio {dataset_path}")
        return
    
    print("Loading dataset...")
    dataset = EmotionDataset(root=dataset_path)
    
    dataset.print_info()
    dataset.plot_histogram()


if __name__ == "__main__":
    main()