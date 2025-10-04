"""
Script de utilidad para gestión de experimentos y configuraciones
"""
import sys
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_manager import ConfigManager


def list_experiments():
    """Lista todos los experimentos disponibles"""
    try:
        config_manager = ConfigManager()
        experiments = config_manager.list_experiments()
        
        if not experiments:
            print("No se encontraron experimentos guardados.")
            return
        
        print("=== EXPERIMENTOS DISPONIBLES ===")
        print(f"{'Timestamp':<20} {'Experiment ID':<20} {'Config Hash':<12} {'Descripción'}")
        print("-" * 80)
        
        for exp in experiments:
            timestamp = exp['timestamp']
            exp_id = exp['experiment_id'][:18] + "..." if len(exp['experiment_id']) > 18 else exp['experiment_id']
            config_hash = exp['config_hash']
            description = exp['description'][:30] + "..." if len(exp['description']) > 30 else exp['description']
            
            print(f"{timestamp:<20} {exp_id:<20} {config_hash:<12} {description}")
            
    except Exception as e:
        print(f"Error al listar experimentos: {e}")


def show_experiment_details(experiment_id: str, config_hash: str):
    """Muestra los detalles de un experimento específico"""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config_from_experiment(experiment_id, config_hash)
        
        print(f"=== DETALLES DEL EXPERIMENTO ===")
        print(f"Experiment ID: {experiment_id}")
        print(f"Config Hash: {config_hash}")
        
        # Información de versión
        version_info = config.get('version_control', {})
        print(f"\nVersión del proyecto: {version_info.get('version', 'N/A')}")
        print(f"Descripción: {version_info.get('description', 'N/A')}")
        print(f"Autor: {version_info.get('author', 'N/A')}")
        
        # Configuración del modelo
        model_config = config.get('model', {})
        print(f"\n--- CONFIGURACIÓN DEL MODELO ---")
        print(f"Arquitectura: {model_config.get('architecture', 'N/A')}")
        print(f"Clases: {model_config.get('num_classes', 'N/A')}")
        print(f"Dropout: {model_config.get('dropout_prob', 'N/A')}")
        
        # Configuración de entrenamiento
        training_config = config.get('training', {})
        print(f"\n--- CONFIGURACIÓN DE ENTRENAMIENTO ---")
        print(f"Épocas: {training_config.get('num_epochs', 'N/A')}")
        print(f"Batch size: {training_config.get('batch_size', 'N/A')}")
        print(f"Learning rate: {training_config.get('learning_rate', 'N/A')}")
        print(f"Weight decay: {training_config.get('weight_decay', 'N/A')}")
        
        # Early stopping
        early_stopping = training_config.get('early_stopping', {})
        if early_stopping.get('enabled', False):
            print(f"Early stopping: Habilitado (paciencia: {early_stopping.get('patience', 'N/A')})")
        
        # Configuración de datos
        data_config = config.get('data', {})
        print(f"\n--- CONFIGURACIÓN DE DATOS ---")
        print(f"Tamaño de imagen: {data_config.get('image_size', 'N/A')}")
        print(f"Muestras por clase: {data_config.get('target_samples_per_class', 'N/A')}")
        print(f"Split validación: {data_config.get('validation_split', 'N/A')}")
        
    except Exception as e:
        print(f"Error al mostrar detalles del experimento: {e}")


def compare_experiments(exp1_id: str, hash1: str, exp2_id: str, hash2: str):
    """Compara dos experimentos"""
    try:
        config_manager = ConfigManager()
        
        config1 = config_manager.load_config_from_experiment(exp1_id, hash1)
        config2 = config_manager.load_config_from_experiment(exp2_id, hash2)
        
        print("=== COMPARACIÓN DE EXPERIMENTOS ===")
        print(f"Experimento 1: {exp1_id} ({hash1})")
        print(f"Experimento 2: {exp2_id} ({hash2})")
        
        # Comparar configuraciones principales
        sections_to_compare = ['model', 'training', 'data']
        
        for section in sections_to_compare:
            print(f"\n--- {section.upper()} ---")
            
            config1_section = config1.get(section, {})
            config2_section = config2.get(section, {})
            
            # Encontrar todas las claves
            all_keys = set(config1_section.keys()) | set(config2_section.keys())
            
            for key in sorted(all_keys):
                val1 = config1_section.get(key, 'N/A')
                val2 = config2_section.get(key, 'N/A')
                
                if val1 != val2:
                    print(f"  {key}:")
                    print(f"    Exp1: {val1}")
                    print(f"    Exp2: {val2}")
                    print(f"    DIFERENTE ⚠️")
                else:
                    print(f"  {key}: {val1} ")
                    
    except Exception as e:
        print(f"Error al comparar experimentos: {e}")


def show_current_config():
    """Muestra la configuración actual"""
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print("=== CONFIGURACIÓN ACTUAL ===")
        
        # Información de versión
        version_info = config.get('version_control', {})
        print(f"Proyecto: {version_info.get('project_name', 'N/A')}")
        print(f"Versión: {version_info.get('version', 'N/A')}")
        print(f"Descripción: {version_info.get('description', 'N/A')}")
        
        # Configuración del modelo
        model_config = config.get('model', {})
        print(f"\n--- MODELO ---")
        for key, value in model_config.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
        
        # Configuración de entrenamiento
        training_config = config.get('training', {})
        print(f"\n--- ENTRENAMIENTO ---")
        for key, value in training_config.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
        
        # Configuración de datos
        data_config = config.get('data', {})
        print(f"\n--- DATOS ---")
        for key, value in data_config.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error al mostrar configuración actual: {e}")


def create_config_backup():
    """Crea un backup de la configuración actual"""
    try:
        config_manager = ConfigManager()
        
        # Crear backup con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("config/backups")
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"config_backup_{timestamp}.yaml"
        
        import shutil
        shutil.copy2(config_manager.config_path, backup_file)
        
        print(f" Backup de configuración creado: {backup_file}")
        
    except Exception as e:
        print(f"Error al crear backup: {e}")


def main():
    """Función principal del script de utilidades"""
    parser = argparse.ArgumentParser(description="Gestión de experimentos y configuraciones")
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando para listar experimentos
    list_parser = subparsers.add_parser('list', help='Lista todos los experimentos')
    
    # Comando para mostrar detalles
    details_parser = subparsers.add_parser('details', help='Muestra detalles de un experimento')
    details_parser.add_argument('experiment_id', help='ID del experimento')
    details_parser.add_argument('config_hash', help='Hash de la configuración')
    
    # Comando para comparar experimentos
    compare_parser = subparsers.add_parser('compare', help='Compara dos experimentos')
    compare_parser.add_argument('exp1_id', help='ID del primer experimento')
    compare_parser.add_argument('hash1', help='Hash del primer experimento')
    compare_parser.add_argument('exp2_id', help='ID del segundo experimento')
    compare_parser.add_argument('hash2', help='Hash del segundo experimento')
    
    # Comando para mostrar configuración actual
    current_parser = subparsers.add_parser('current', help='Muestra la configuración actual')
    
    # Comando para crear backup
    backup_parser = subparsers.add_parser('backup', help='Crea un backup de la configuración actual')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments()
    elif args.command == 'details':
        show_experiment_details(args.experiment_id, args.config_hash)
    elif args.command == 'compare':
        compare_experiments(args.exp1_id, args.hash1, args.exp2_id, args.hash2)
    elif args.command == 'current':
        show_current_config()
    elif args.command == 'backup':
        create_config_backup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()