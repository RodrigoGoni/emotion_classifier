"""Script simplificado para evaluar el modelo entrenado"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluate import Evaluator
from src.utils.config_manager import ConfigManager


def main():
    """Función principal simplificada"""
    print("=== EVALUADOR DE MODELO ===")
    
    try:
        config_manager = ConfigManager()
        print("ConfigManager inicializado")
        
        # Configurar device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando device: {device}")
        
        # Directorio de resultados
        results_dir = config_manager.get_results_dir() / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Buscar modelo más reciente
        models_dir = Path(config_manager.get_config('output')['models_dir'])
        model_files = list(models_dir.glob("*.pth"))
        
        if not model_files:
            print("No se encontraron modelos entrenados")
            return None
        
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Evaluando modelo: {model_path.name}")
        
        # Cargar dataset de validación
        data_config = config_manager.get_config('data')
        val_data_path = Path(data_config['val_path'])
        
        if not val_data_path.exists():
            print(f"Error: {val_data_path} no existe")
            print("Ejecuta primero: python scripts/create_balanced_dataset.py")
            return None
        
        val_dataset = EmotionDataset(str(val_data_path), transform=get_val_transforms())
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        print(f"Dataset de validación: {len(val_dataset)} muestras")
        
        # Cargar modelo
        model, checkpoint = Evaluator.load_model(str(model_path), device)
        print(f"Modelo cargado - Época: {checkpoint['epoch']}, Val Accuracy: {checkpoint['val_acc']:.4f}")
        
        # Crear evaluador y generar reporte completo
        evaluator = Evaluator(model, val_loader, device)
        print("Generando reporte completo...")
        
        report_path, json_path = evaluator.generate_complete_report(config_manager, model_path, results_dir)
        
        print(f"\nEVALUACIÓN COMPLETADA")
        print(f"Reporte completo: {report_path}")
        print(f"Datos JSON: {json_path}")
        print(f"Visualizaciones en: {results_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()