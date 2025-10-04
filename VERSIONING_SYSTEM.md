# Sistema de Gestión de Versiones y Configuraciones

## Descripción

Este sistema proporciona un manejo centralizado de configuraciones y versionado automático para experimentos de clasificación de emociones. Permite realizar un seguimiento completo de los experimentos, configuraciones y resultados.

## Características Principales

### 1. Configuración Centralizada
- **Archivo único**: `config/config.yaml` contiene toda la configuración del proyecto
- **Secciones organizadas**: Modelo, entrenamiento, datos, evaluación, tracking
- **Versionado automático**: Cada experimento guarda su configuración específica

### 2. Gestión de Experimentos
- **MLflow integrado**: Tracking automático de métricas y parámetros
- **Versionado de modelos**: Nombres únicos con timestamp y hash de configuración
- **Metadatos completos**: Información de experimento, configuración y resultados

### 3. Organización de Resultados
- **Directorios versionados**: Cada experimento tiene su propio directorio
- **Resúmenes automáticos**: JSON con toda la información del experimento
- **Backups de configuración**: Sistema de respaldo automático

## Estructura de Archivos

```
config/
├── config.yaml           # Configuración principal
├── versions/             # Versiones de configuraciones por experimento
└── backups/             # Backups de configuraciones

results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── evaluation/
    │   ├── confusion_matrix.png
    │   ├── class_accuracy.png
    │   ├── classification_report.txt
    │   └── evaluation_results.json
    └── experiment_summary.json

models/trained/
└── CNN_Custom_v1.0.0_YYYYMMDD_HHMMSS_best.pth
```

## Uso del Sistema

### 1. Configuración del Proyecto

Edita `config/config.yaml` para ajustar:

```yaml
# Información del proyecto
version_control:
  project_name: "emotion_classifier"
  version: "1.0.0"
  description: "Tu descripción del experimento"

# Configuración del modelo
model:
  architecture: "CNN_Custom"
  dropout_prob: 0.5

# Configuración de entrenamiento
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### 2. Scripts Principales

#### Crear Dataset Balanceado
```bash
python scripts/create_balanced_dataset.py
```
- Usa configuración de `data` section
- Crea splits train/val automáticamente
- Guarda metadatos del dataset

#### Entrenar Modelo
```bash
python scripts/train_model.py
```
- Configura experimento MLflow automáticamente
- Guarda modelo con versionado
- Crea resumen del experimento

#### Evaluar Modelo
```bash
python scripts/evaluate_model.py
```
- Evalúa el modelo más reciente
- Genera visualizaciones y reportes
- Guarda resultados versionados

### 3. Gestión de Experimentos

#### Ver configuración actual:
```bash
python scripts/manage_experiments.py current
```

#### Listar todos los experimentos:
```bash
python scripts/manage_experiments.py list
```

#### Ver detalles de un experimento:
```bash
python scripts/manage_experiments.py details EXPERIMENT_ID CONFIG_HASH
```

#### Comparar dos experimentos:
```bash
python scripts/manage_experiments.py compare EXP1_ID HASH1 EXP2_ID HASH2
```

#### Crear backup de configuración:
```bash
python scripts/manage_experiments.py backup
```

## Configuraciones Disponibles

### Datos (`data`)
- `target_samples_per_class`: Muestras por clase en dataset balanceado
- `validation_split`: Proporción para validación
- `image_size`: Tamaño de imágenes
- `augmentation`: Configuración de data augmentation

### Modelo (`model`)
- `architecture`: Tipo de arquitectura
- `dropout_prob`: Probabilidad de dropout
- `conv_layers`: Configuración de capas convolucionales
- `fc_layers`: Configuración de capas fully connected

### Entrenamiento (`training`)
- `num_epochs`: Número de épocas
- `batch_size`: Tamaño de batch
- `learning_rate`: Tasa de aprendizaje
- `early_stopping`: Configuración de early stopping
- `scheduler`: Configuración del scheduler

### Evaluación (`evaluation`)
- `metrics`: Métricas a calcular
- `test_datasets`: Datasets de prueba
- `save_predictions`: Guardar predicciones

### Tracking (`tracking`)
- `mlflow_uri`: URI de MLflow
- `log_model`: Si logear el modelo
- `metrics_to_log`: Métricas a trackear

## Versionado Automático

### Modelos
Los modelos se guardan con el formato:
```
{architecture}_v{version}_{timestamp}_{tipo}.pth
```

Ejemplo: `CNN_Custom_v1.0.0_20251004_143022_best.pth`

### Experimentos
Cada experimento genera:
1. **Run MLflow**: Con métricas y parámetros
2. **Configuración versionada**: En `config/versions/`
3. **Directorio de resultados**: Con timestamp único
4. **Resumen del experimento**: JSON con toda la información

### Hash de Configuración
Se genera un hash MD5 de la configuración para identificar cambios:
- Mismo hash = misma configuración
- Hash diferente = configuración modificada

## Ejemplo de Flujo Completo

```bash
# 1. Configurar experimento
nano config/config.yaml

# 2. Crear dataset balanceado
python scripts/create_balanced_dataset.py

# 3. Entrenar modelo
python scripts/train_model.py

# 4. Evaluar modelo
python scripts/evaluate_model.py

# 5. Ver resultados
python scripts/manage_experiments.py list
python scripts/manage_experiments.py details EXPERIMENT_ID CONFIG_HASH
```

## Solución de Problemas

### Error de importación
Si aparecen errores de importación del ConfigManager:
```bash
export PYTHONPATH=/home/rodrigo/Desktop/maestria/emotion_classifier:$PYTHONPATH
```

### Configuración no encontrada
Verificar que existe `config/config.yaml` y tiene el formato correcto.

### MLflow no funciona
Verificar que el directorio `mlruns` existe y tiene permisos de escritura.