"""
Script de entrenamiento con MLflow para clasificación de emociones
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.cnn_model import CNNModel
from src.training.train import Trainner
from src.utils.constants import *


class EarlyStopping:
    """Early stopping para detener el entrenamiento cuando no mejora la validación"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Guarda el mejor modelo"""
        self.best_weights = model.state_dict().copy()


def setup_mlflow():
    """Configurar MLflow"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("emotion_classification")


def train_model(config):
    """Entrenamiento principal con MLflow"""
    
    with mlflow.start_run():
        # Log de parámetros
        mlflow.log_params(config)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Datasets
        train_dataset = EmotionDataset(
            root=config['train_data_path'],
            transform=get_train_transforms()
        )
        val_dataset = EmotionDataset(
            root=config['val_data_path'],
            transform=get_val_transforms()
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Modelo
        model = CNNModel(
            num_classes=NUM_CLASSES,
            input_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_channels=CHANNELS,
            dropout_prob=config['dropout_prob']
        ).to(device)
        
        # Criterio y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
        
        # Trainer
        trainer = Trainner(model, train_loader, val_loader, criterion, optimizer, device)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config['patience'],
            min_delta=config['min_delta'],
            restore_best_weights=True
        )
        
        # Training loop
        best_val_acc = 0.0
        best_model_path = None
        
        for epoch in range(config['num_epochs']):
            # Train
            train_loss = trainer.train_epoch()
            
            # Validate
            val_loss, val_acc = trainer.validate_epoch()
            
            # Scheduler step
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log métricas
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': current_lr
            }, step=epoch)
            
            print(f"Epoch {epoch+1}: Train={train_loss:.3f}, Val={val_loss:.3f}, Val Acc={val_acc:.3f}")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = f"models/trained/best_model_epoch_{epoch+1}.pth"
                
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': config
                }, best_model_path)
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Log del mejor modelo
        mlflow.log_metric('best_val_accuracy', best_val_acc)
        
        if best_model_path and os.path.exists(best_model_path):
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(best_model_path, "best_model")
        
        print(f"Entrenamiento completado. Mejor accuracy: {best_val_acc:.4f}")
        
        return model, best_val_acc


def main():
    """Función principal"""
    setup_mlflow()
    
    # Configuración de entrenamiento
    config = {
        'train_data_path': 'data/processed/train',
        'val_data_path': 'data/processed/val',
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'dropout_prob': 0.5,
        'step_size': 15,
        'gamma': 0.5,
        'patience': 10,
        'min_delta': 0.001,
        'model_type': 'CNN_Custom'
    }
    
    # Verificar que existen los datos
    if not Path(config['train_data_path']).exists():
        print(f"Error: {config['train_data_path']} no existe")
        return
    
    print("Iniciando entrenamiento...")
    
    # Entrenar modelo
    model, best_acc = train_model(config)
    
    print(f"Finalizado. Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
