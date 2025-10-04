"""
Constantes utilizadas en todo el proyecto
Ahora se obtienen del ConfigManager para un manejo centralizado
"""
import os
from pathlib import Path

# Constantes básicas que no cambian
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"

# Función para obtener configuración de manera lazy
def get_config():
    """Obtiene la configuración usando el ConfigManager"""
    from .config_manager import ConfigManager
    return ConfigManager(str(CONFIG_PATH))

# Emociones y clases - estas son constantes del dominio
EMOTION_CLASSES = {
    0: "alegria",
    1: "disgusto", 
    2: "enojo",
    3: "miedo",
    4: "seriedad",
    5: "sorpresa",
    6: "tristeza"
}

EMOTION_LABELS = list(EMOTION_CLASSES.values())
NUM_CLASSES = len(EMOTION_CLASSES)

# Valores por defecto para compatibilidad con código existente
# Estos ahora se obtienen de la configuración pero mantenemos valores por defecto
try:
    config_manager = get_config()
    data_config = config_manager.get_config('data')
    model_config = config_manager.get_config('model')
    training_config = config_manager.get_config('training')
    
    # Configuración de imágenes
    IMAGE_SIZE = data_config.get('image_size', 100)
    CHANNELS = data_config.get('channels', 3)
    
    # Configuración de entrenamiento por defecto
    DEFAULT_BATCH_SIZE = training_config.get('batch_size', 32)
    DEFAULT_LEARNING_RATE = training_config.get('learning_rate', 0.001)
    DEFAULT_NUM_EPOCHS = training_config.get('num_epochs', 50)
    
    # Normalización ImageNet
    augmentation_config = data_config.get('augmentation', {})
    if augmentation_config.get('normalize_imagenet', True):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    else:
        IMAGENET_MEAN = [0.5, 0.5, 0.5]
        IMAGENET_STD = [0.5, 0.5, 0.5]
    
except Exception:
    # Valores por defecto si no se puede cargar la configuración
    IMAGE_SIZE = 100
    CHANNELS = 3
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_NUM_EPOCHS = 50
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# Rutas por defecto - se mantienen para compatibilidad
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_MODELS_DIR = "models/trained"
DEFAULT_RESULTS_DIR = "results"

# Configuración de detección de rostros
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 6

# Mapeo de AffectNet a nuestro modelo
AFFECTNET_CLASSES = {
    0: "Anger",
    1: "Disgust", 
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
    7: "Contempt"
}

AFFECTNET_TO_YOUR_MODEL = {
    # AffectNet ID: Tu Modelo ID
    3: 0,  # Happy -> alegria
    1: 1,  # Disgust -> disgusto
    0: 2,  # Anger -> enojo
    2: 3,  # Fear -> miedo
    6: 4,  # Neutral -> seriedad
    5: 5,  # Surprise -> sorpresa
    4: 6,  # Sad -> tristeza
    7: None  # Contempt -> EXCLUIR
}
