"""
Script para evaluar el modelo entrenado
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluate import Evaluator
from src.utils.constants import *



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


def main():
    """Función principal de evaluación"""
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Rutas
    val_data_path = "data/processed/val"
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar el mejor modelo
    models_dir = Path("models/trained")
    model_files = list(models_dir.glob("best_model_epoch_*.pth"))
    
    if not model_files:
        print("Error: No se encontraron modelos entrenados")
        return
    
    # Tomar el modelo más reciente
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Evaluando modelo: {latest_model}")
    
    # Cargar dataset de validación
    if not Path(val_data_path).exists():
        print(f"Error: {val_data_path} no existe")
        print("Ejecuta primero: python scripts/create_balanced_dataset.py")
        return
    
    val_dataset = EmotionDataset(val_data_path, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    # Cargar modelo usando la clase Evaluator
    model, checkpoint = Evaluator.load_model(latest_model, device)
    
    # Crear evaluador
    evaluator = Evaluator(model, val_loader, device)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    results = evaluator.get_detailed_results()
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    class_names = results['class_names']
    
    # Generar matriz de confusión
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, predictions, class_names, confusion_matrix_path)
    
    # Generar gráfico de accuracy por clase
    class_accuracy_path = results_dir / "class_accuracy.png"
    plot_class_accuracy(true_labels, predictions, class_names, class_accuracy_path)
    
    # Generar reporte de clasificación
    report_path = results_dir / "classification_report.txt"
    generate_classification_report(true_labels, predictions, class_names, report_path)
    
    # Mostrar accuracy general
    print(f"ACCURACY GENERAL: {results['overall_accuracy']:.4f}")
    
    # Mostrar accuracy por clase
    print("\nAccuracy por clase:")
    for emotion, acc in results['class_accuracies'].items():
        print(f"  {emotion}: {acc:.4f}")
    
    print(f"\nResultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()
