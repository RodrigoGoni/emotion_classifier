"""
Script de entrenamiento con MLflow y gestión de versiones para clasificación de emociones
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
from src.utils.constants import NUM_CLASSES
from src.utils.config_manager import ConfigManager


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


def train_model(config_manager: ConfigManager):
    """Entrenamiento principal con MLflow y versionado"""
    
    # Obtener configuraciones
    data_config = config_manager.get_config('data')
    model_config = config_manager.get_config('model')
    training_config = config_manager.get_config('training')
    
    # Configurar experimento
    experiment_id = config_manager.setup_mlflow_experiment()
    run_id = config_manager.start_mlflow_run()
    
    # Guardar versión de configuración
    config_hash = config_manager.save_config_version(
        experiment_id, 
        f"Entrenamiento {model_config['architecture']} con {training_config['num_epochs']} épocas"
    )
    
    print(f"Experimento iniciado: {experiment_id}")
    print(f"Run ID: {run_id}")
    print(f"Config Hash: {config_hash}")
    
    # Setup device
    device_config = training_config.get('device', 'auto')
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    print(f"Usando device: {device}")
    
    # Datasets
    train_dataset = EmotionDataset(
        root=data_config['train_path'],
        transform=get_train_transforms()
    )
    val_dataset = EmotionDataset(
        root=data_config['val_path'],
        transform=get_val_transforms()
    )
    
    print(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True)
    )
    
    # Modelo
    model = CNNModel(
        num_classes=NUM_CLASSES,
        input_size=(data_config['image_size'], data_config['image_size']),
        num_channels=data_config['channels'],
        dropout_prob=model_config['dropout_prob']
    ).to(device)
    
    print(f"Modelo {model_config['architecture']} creado con {NUM_CLASSES} clases")
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    
    optimizer_config = training_config.get('optimizer', {})
    if optimizer_config.get('type', 'Adam') == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    
    # Scheduler
    scheduler_config = training_config.get('scheduler', {})
    if scheduler_config.get('type', 'StepLR') == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 15),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Trainer
    trainer = Trainner(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = None
    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True)
        )
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = None
    
    print("Iniciando entrenamiento...")
    
    for epoch in range(training_config['num_epochs']):
        # Train
        train_loss = trainer.train_epoch()
        
        # Validate
        val_loss, val_acc = trainer.validate_epoch()
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log métricas
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': current_lr,
            'epoch': epoch + 1
        }
        
        # Log a MLflow
        tracking_config = config_manager.get_config('tracking')
        if tracking_config.get('log_metrics', True):
            mlflow.log_metrics(metrics, step=epoch)
        
        print(f"Epoch {epoch+1}/{training_config['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = config_manager.get_model_save_path(epoch + 1, is_best=True)
            
            # Asegurar que el directorio existe
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar modelo con metadatos completos
            model_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config_hash': config_hash,
                'experiment_id': experiment_id,
                'run_id': run_id,
                'model_config': model_config,
                'training_config': training_config,
                'data_config': data_config
            }
            
            torch.save(model_checkpoint, best_model_path)
            print(f"Nuevo mejor modelo guardado: {best_model_path}")
        
        # Early stopping check
        if early_stopping and early_stopping(val_loss, model):
            print(f"Early stopping activado en época {epoch+1}")
            break
    
    # Log del mejor modelo y métricas finales
    final_metrics = {
        'best_val_accuracy': best_val_acc,
        'total_epochs_trained': epoch + 1,
        'config_hash': config_hash
    }
    
    mlflow.log_metrics(final_metrics)
    
    # Log del modelo en MLflow
    tracking_config = config_manager.get_config('tracking')
    if tracking_config.get('log_model', True) and best_model_path and best_model_path.exists():
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(str(best_model_path), "best_model")
    
    # Guardar resumen del experimento
    config_manager.save_experiment_summary(
        final_metrics, 
        str(best_model_path),
        {"total_parameters": sum(p.numel() for p in model.parameters())}
    )
    
    print(f"Entrenamiento completado!")
    print(f"Mejor accuracy de validación: {best_val_acc:.4f}")
    print(f"Modelo guardado en: {best_model_path}")
    
    return model, best_val_acc, config_hash


def main():
    """Función principal"""
    print("=== ENTRENAMIENTO DE MODELO DE EMOCIONES ===")
    print("Iniciando sistema de gestión de configuraciones...")
    
    # Inicializar ConfigManager
    try:
        config_manager = ConfigManager()
        print(" ConfigManager inicializado correctamente")
    except Exception as e:
        print(f" Error al inicializar ConfigManager: {e}")
        return
    
    # Mostrar información del experimento
    version_info = config_manager.get_config('version_control')
    print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
    print(f"Descripción: {version_info['description']}")
    print(f"Autor: {version_info['author']}")
    
    # Verificar configuración de datos
    data_config = config_manager.get_config('data')
    train_path = Path(data_config['train_path'])
    val_path = Path(data_config['val_path'])
    
    if not train_path.exists():
        print(f" Error: {train_path} no existe")
        print("Por favor, ejecuta el script de creación de dataset balanceado primero")
        return
        
    if not val_path.exists():
        print(f" Error: {val_path} no existe")
        print("Por favor, ejecuta el script de creación de dataset balanceado primero")
        return
    
    print(f" Datos de entrenamiento encontrados en: {train_path}")
    print(f" Datos de validación encontrados en: {val_path}")
    
    # Mostrar configuración principal
    training_config = config_manager.get_config('training')
    model_config = config_manager.get_config('model')
    
    print("\n=== CONFIGURACIÓN DEL EXPERIMENTO ===")
    print(f"Arquitectura: {model_config['architecture']}")
    print(f"Épocas: {training_config['num_epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Dropout: {model_config['dropout_prob']}")
    
    early_stopping_config = training_config.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        print(f"Early stopping: Habilitado (paciencia: {early_stopping_config.get('patience', 10)})")
    else:
        print("Early stopping: Deshabilitado")
    
    # Confirmar inicio
    print("\n=== INICIANDO ENTRENAMIENTO ===")
    
    try:
        # Entrenar modelo
        model, best_acc, config_hash = train_model(config_manager)
        
        print("\n=== ENTRENAMIENTO COMPLETADO ===")
        print(f" Mejor accuracy de validación: {best_acc:.4f}")
        print(f" Hash de configuración: {config_hash}")
        
        # Mostrar información de versionado
        results_dir = config_manager.get_results_dir(create=False)
        print(f" Resultados guardados en: {results_dir}")
        
        # Finalizar run de MLflow
        if mlflow.active_run():
            mlflow.end_run()
            print(" Run de MLflow finalizado")
        
    except Exception as e:
        print(f" Error durante el entrenamiento: {e}")
        # Finalizar run en caso de error
        if mlflow.active_run():
            mlflow.end_run(status='FAILED')
        raise


if __name__ == "__main__":
    main()
