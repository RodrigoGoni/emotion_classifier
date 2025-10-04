"""
Módulo de evaluación para modelos entrenados
"""
import torch
import numpy as np
from pathlib import Path
from src.models.cnn_model import CNNModel
from src.utils.constants import *


class Evaluator:
    """Clase para evaluar modelos entrenados"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
    
    @staticmethod
    def load_model(model_path, device):
        """Cargar modelo entrenado desde checkpoint"""
        checkpoint = torch.load(model_path, map_location=device)
        
        model = CNNModel(
            num_classes=NUM_CLASSES,
            input_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_channels=CHANNELS,
            dropout_prob=0.5
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Modelo cargado desde: {model_path}")
        print(f"Época: {checkpoint['epoch']}")
        print(f"Val Accuracy: {checkpoint['val_acc']:.4f}")
        
        return model, checkpoint
    
    def evaluate_model(self):
        """Evaluar modelo y obtener predicciones y etiquetas verdaderas"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities)
        }
    
    def get_accuracy(self):
        """Calcular accuracy del modelo"""
        results = self.evaluate_model()
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        correct = (predictions == true_labels).sum()
        total = len(true_labels)
        accuracy = correct / total
        
        return accuracy
    
    def get_class_accuracies(self):
        """Calcular accuracy por cada clase"""
        results = self.evaluate_model()
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        class_accuracies = {}
        for class_idx in range(NUM_CLASSES):
            class_mask = (true_labels == class_idx)
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_true = true_labels[class_mask]
                class_accuracy = (class_predictions == class_true).sum() / len(class_true)
                class_accuracies[EMOTION_CLASSES[class_idx]] = class_accuracy
            else:
                class_accuracies[EMOTION_CLASSES[class_idx]] = 0.0
        
        return class_accuracies
    
    def get_detailed_results(self):
        """Obtener resultados detallados para análisis posterior"""
        results = self.evaluate_model()
        
        return {
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'overall_accuracy': self.get_accuracy(),
            'class_accuracies': self.get_class_accuracies(),
            'num_samples': len(results['true_labels']),
            'num_classes': NUM_CLASSES,
            'class_names': [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
        }