"""
Script para evaluar el modelo entrenado con gestión de versiones
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluate import Evaluator
from src.utils.constants import EMOTION_LABELS, NUM_CLASSES
from src.utils.config_manager import ConfigManager



def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generar y guardar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Matriz de confusión guardada: {save_path}")


def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Generar gráfico de accuracy por clase"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy, color='skyblue')
    plt.title('Accuracy por Clase')
    plt.xlabel('Emociones')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Accuracy por clase guardado: {save_path}")


def generate_classification_report(y_true, y_pred, class_names, save_path):
    """Generar y guardar reporte de clasificación"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(save_path, 'w') as f:
        f.write("REPORTE DE CLASIFICACIÓN\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nACCURACY GENERAL\n")
        f.write("=" * 20 + "\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    
    print(f"Reporte de clasificación guardado: {save_path}")
    print("\nReporte de Clasificación:")
    print(report)


def evaluate_model_with_config(config_manager: ConfigManager, model_path: str = None):
    """Evalúa un modelo usando la configuración centralizada"""
    
    # Obtener configuraciones
    data_config = config_manager.get_config('data')
    evaluation_config = config_manager.get_config('evaluation')
    version_info = config_manager.get_config('version_control')
    
    print(f"=== EVALUACIÓN DE MODELO ===")
    print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Directorio de resultados versionado
    results_dir = config_manager.get_results_dir() / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar modelo a evaluar
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"✗ Error: Modelo no encontrado en {model_path}")
            return None
    else:
        # Buscar el modelo más reciente
        models_dir = Path(config_manager.get_config('output')['models_dir'])
        model_files = list(models_dir.glob("*.pth"))
        
        if not model_files:
            print(" Error: No se encontraron modelos entrenados")
            return None
        
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Evaluando modelo: {model_path}")
    
    # Cargar dataset de validación
    val_data_path = Path(data_config['val_path'])
    if not val_data_path.exists():
        print(f" Error: {val_data_path} no existe")
        print("Ejecuta primero: python scripts/create_balanced_dataset.py")
        return None
    
    val_dataset = EmotionDataset(str(val_data_path), transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    # Cargar modelo usando la clase Evaluator
    try:
        model, checkpoint = Evaluator.load_model(str(model_path), device)
        print(" Modelo cargado exitosamente")
        
        # Obtener información del checkpoint si está disponible
        model_info = {}
        if checkpoint:
            model_info = {
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_acc': checkpoint.get('val_acc', 'N/A'),
                'config_hash': checkpoint.get('config_hash', 'N/A'),
                'experiment_id': checkpoint.get('experiment_id', 'N/A')
            }
            print(f"  Época: {model_info['epoch']}")
            if isinstance(model_info['val_acc'], float):
                print(f"  Val Accuracy (entrenamiento): {model_info['val_acc']:.4f}")
            print(f"  Config Hash: {model_info['config_hash']}")
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        return None
    
    # Crear evaluador
    evaluator = Evaluator(model, val_loader, device)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    results = evaluator.get_detailed_results()
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    class_names = results['class_names']
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    
    # Matriz de confusión
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, predictions, class_names, confusion_matrix_path)
    
    # Accuracy por clase
    class_accuracy_path = results_dir / "class_accuracy.png"
    plot_class_accuracy(true_labels, predictions, class_names, class_accuracy_path)
    
    # Reporte de clasificación
    report_path = results_dir / "classification_report.txt"
    generate_classification_report(true_labels, predictions, class_names, report_path)
    
    # Calcular métricas adicionales
    overall_accuracy = results['overall_accuracy']
    class_accuracies = results['class_accuracies']
    
    # Guardar resultados completos
    evaluation_results = {
        "timestamp": config_manager.timestamp,
        "model_info": {
            "model_path": str(model_path),
            "checkpoint_info": model_info
        },
        "dataset_info": {
            "val_path": str(val_data_path),
            "total_samples": len(val_dataset),
            "num_classes": NUM_CLASSES
        },
        "metrics": {
            "overall_accuracy": float(overall_accuracy),
            "class_accuracies": {k: float(v) for k, v in class_accuracies.items()}
        },
        "evaluation_config": evaluation_config,
        "predictions_summary": {
            "total_predictions": len(predictions),
            "correct_predictions": sum(p == t for p, t in zip(predictions, true_labels))
        }
    }
    
    # Guardar resultados en JSON
    results_json_path = results_dir / "evaluation_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print(f"\n=== RESULTADOS DE EVALUACIÓN ===")
    print(f" Accuracy general: {overall_accuracy:.4f}")
    print(f" Muestras evaluadas: {len(predictions)}")
    print(f" Predicciones correctas: {evaluation_results['predictions_summary']['correct_predictions']}")
    
    print("\nAccuracy por clase:")
    for emotion, acc in class_accuracies.items():
        print(f"  {emotion}: {acc:.4f}")
    
    print(f"\n Resultados guardados en: {results_dir}")
    print(f" Archivo de resultados: {results_json_path}")
    
    return evaluation_results


def main():
    """Función principal"""
    print("=== EVALUADOR DE MODELO ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager()
        print(" ConfigManager inicializado")
        
        # Evaluar modelo
        results = evaluate_model_with_config(config_manager)
        
        if results:
            print("\n Evaluación completada exitosamente")
        else:
            print("\n Evaluación falló")
            
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
