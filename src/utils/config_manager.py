"""
Gestor de configuraciones y versionado para el proyecto de clasificación de emociones
"""
import os
import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow


class ConfigManager:
    """Gestor centralizado de configuraciones y versionado"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ConfigManager
        
        Args:
            config_path: Ruta al archivo de configuración principal
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene la configuración completa o una sección específica
        
        Args:
            section: Sección específica a obtener (ej: 'training', 'model')
            
        Returns:
            Diccionario con la configuración
        """
        if section:
            return self.config.get(section, {})
        return self.config.copy()
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Actualiza una sección de la configuración
        
        Args:
            section: Sección a actualizar
            updates: Diccionario con las actualizaciones
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section].update(updates)
    
    def save_config_version(self, experiment_id: str, description: str = "") -> str:
        """
        Guarda una versión de la configuración con metadatos
        
        Args:
            experiment_id: ID del experimento
            description: Descripción opcional del experimento
            
        Returns:
            Hash de la configuración para versionado
        """
        # Crear hash de la configuración
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Crear directorio de versiones
        versions_dir = Path("config/versions")
        versions_dir.mkdir(exist_ok=True)
        
        # Metadatos de la versión
        version_metadata = {
            "experiment_id": experiment_id,
            "config_hash": config_hash,
            "timestamp": self.timestamp,
            "description": description,
            "config": self.config
        }
        
        # Guardar versión
        version_file = versions_dir / f"config_{experiment_id}_{config_hash}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_metadata, f, indent=2, ensure_ascii=False)
            
        return config_hash
    
    def setup_mlflow_experiment(self) -> str:
        """
        Configura el experimento de MLflow con versionado
        
        Returns:
            ID del experimento
        """
        # Configurar MLflow
        mlflow_config = self.get_config('tracking')
        mlflow.set_tracking_uri(mlflow_config['mlflow_uri'])
        
        # Crear nombre del experimento con timestamp
        experiment_name = self.config['version_control']['experiment_name']
        experiment_name_with_version = f"{experiment_name}_{self.timestamp}"
        
        # Configurar experimento
        experiment = mlflow.set_experiment(experiment_name_with_version)
        experiment_id = experiment.experiment_id
        
        return experiment_id
    
    def start_mlflow_run(self, run_name: Optional[str] = None) -> str:
        """
        Inicia un run de MLflow con configuración automática
        
        Args:
            run_name: Nombre opcional del run
            
        Returns:
            ID del run
        """
        if not run_name:
            version_info = self.get_config('version_control')
            model_config = self.get_config('model')
            run_name = f"{model_config['architecture']}_v{version_info['version']}_{self.timestamp}"
        
        # Iniciar run
        mlflow_run = mlflow.start_run(run_name=run_name)
        self.run_id = mlflow_run.info.run_id
        
        # Log de parámetros principales
        self._log_config_to_mlflow()
        
        return self.run_id
    
    def _log_config_to_mlflow(self) -> None:
        """Registra la configuración en MLflow"""
        # Log de información de versión
        version_info = self.get_config('version_control')
        mlflow.log_params({
            f"version_{k}": v for k, v in version_info.items() 
            if isinstance(v, (str, int, float))
        })
        
        # Log de configuración del modelo
        model_config = self.get_config('model')
        mlflow.log_params({
            f"model_{k}": v for k, v in model_config.items() 
            if isinstance(v, (str, int, float))
        })
        
        # Log de configuración de entrenamiento
        training_config = self.get_config('training')
        flattened_training = self._flatten_dict(training_config, prefix="training")
        mlflow.log_params(flattened_training)
        
        # Log de configuración de datos
        data_config = self.get_config('data')
        flattened_data = self._flatten_dict(data_config, prefix="data")
        mlflow.log_params(flattened_data)
        
        # Guardar archivo de configuración completo
        config_file = f"config_{self.timestamp}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        mlflow.log_artifact(config_file, "config")
        os.remove(config_file)  # Limpiar archivo temporal
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Aplana un diccionario anidado para MLflow"""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
        return dict(items)
    
    def get_model_save_path(self, epoch: Optional[int] = None, is_best: bool = False) -> Path:
        """
        Genera la ruta para guardar el modelo con versionado
        
        Args:
            epoch: Época actual (opcional)
            is_best: Si es el mejor modelo
            
        Returns:
            Ruta donde guardar el modelo
        """
        output_config = self.get_config('output')
        models_dir = Path(output_config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear nombre del modelo
        version_info = self.get_config('version_control')
        model_config = self.get_config('model')
        
        base_name = f"{model_config['architecture']}_v{version_info['version']}_{self.timestamp}"
        
        if is_best:
            filename = f"{base_name}_best.pth"
        elif epoch is not None:
            filename = f"{base_name}_epoch_{epoch}.pth"
        else:
            filename = f"{base_name}_final.pth"
            
        return models_dir / filename
    
    def get_results_dir(self, create: bool = True) -> Path:
        """
        Obtiene el directorio de resultados versionado
        
        Args:
            create: Si crear el directorio
            
        Returns:
            Ruta del directorio de resultados
        """
        output_config = self.get_config('output')
        results_base = Path(output_config['results_dir'])
        
        # Crear subdirectorio con timestamp
        results_dir = results_base / f"experiment_{self.timestamp}"
        
        if create:
            results_dir.mkdir(parents=True, exist_ok=True)
            
        return results_dir
    
    def save_experiment_summary(self, metrics: Dict[str, float], 
                              model_path: str, additional_info: Dict[str, Any] = None) -> None:
        """
        Guarda un resumen del experimento
        
        Args:
            metrics: Métricas finales del experimento
            model_path: Ruta del modelo guardado
            additional_info: Información adicional
        """
        results_dir = self.get_results_dir()
        
        summary = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "run_id": self.run_id,
                "config_hash": hashlib.md5(
                    json.dumps(self.config, sort_keys=True).encode()
                ).hexdigest()[:8]
            },
            "version_control": self.get_config('version_control'),
            "final_metrics": metrics,
            "model_path": str(model_path),
            "config_snapshot": self.config
        }
        
        if additional_info:
            summary["additional_info"] = additional_info
            
        summary_file = results_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # También guardar en MLflow si está activo
        if mlflow.active_run():
            mlflow.log_artifact(str(summary_file), "summary")
    
    def load_config_from_experiment(self, experiment_id: str, config_hash: str) -> Dict[str, Any]:
        """
        Carga una configuración desde un experimento específico
        
        Args:
            experiment_id: ID del experimento
            config_hash: Hash de la configuración
            
        Returns:
            Configuración del experimento
        """
        version_file = Path(f"config/versions/config_{experiment_id}_{config_hash}.json")
        
        if not version_file.exists():
            raise FileNotFoundError(f"Configuración no encontrada: {version_file}")
            
        with open(version_file, 'r', encoding='utf-8') as f:
            version_data = json.load(f)
            
        return version_data['config']
    
    def list_experiments(self) -> list:
        """Lista todos los experimentos guardados"""
        versions_dir = Path("config/versions")
        if not versions_dir.exists():
            return []
            
        experiments = []
        for config_file in versions_dir.glob("config_*.json"):
            with open(config_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                experiments.append({
                    "experiment_id": metadata["experiment_id"],
                    "config_hash": metadata["config_hash"],
                    "timestamp": metadata["timestamp"],
                    "description": metadata.get("description", ""),
                    "file": str(config_file)
                })
                
        return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)