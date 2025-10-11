"""
Script de entrenamiento principal
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

class Trainner():
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Para tracking de métricas
        self.metrics_history = []

    def save_metrics_to_csv(self, results_dir: Path, filename: str = "training_metrics.csv"):
        """
        Guarda las métricas de entrenamiento en un archivo CSV
        
        Args:
            results_dir: Directorio donde guardar el archivo
            filename: Nombre del archivo CSV
        """
        if not self.metrics_history:
            print("No hay métricas para guardar")
            return None
            
        # Crear DataFrame con las métricas
        df = pd.DataFrame(self.metrics_history)
        
        # Asegurar que el directorio existe
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo CSV
        csv_path = results_dir / filename
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        print(f"Métricas guardadas en: {csv_path}")
        return csv_path
    
    def add_metrics(self, epoch: int, train_loss: float, train_acc: float, train_f1: float,
                   val_loss: float, val_acc: float, val_f1: float, learning_rate: float = None):
        """
        Añade métricas de una época al historial
        
        Args:
            epoch: Número de época
            train_loss: Loss de entrenamiento
            train_acc: Accuracy de entrenamiento  
            train_f1: F1 score de entrenamiento
            val_loss: Loss de validación
            val_acc: Accuracy de validación
            val_f1: F1 score de validación
            learning_rate: Learning rate actual (opcional)
        """
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1
        }
        
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
            
        self.metrics_history.append(metrics)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, accuracy, f1

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, accuracy, f1