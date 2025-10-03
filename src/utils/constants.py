"""
Constantes utilizadas en todo el proyecto
"""

# Emociones y sus índices
EMOTION_CLASSES = {
    0: "angry",
    1: "disgust", 
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

EMOTION_LABELS = list(EMOTION_CLASSES.values())
NUM_CLASSES = len(EMOTION_CLASSES)

# Configuración de imágenes
IMAGE_SIZE = 224
CHANNELS = 3

# Configuración de entrenamiento por defecto
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_EPOCHS = 50

# Rutas por defecto
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_MODELS_DIR = "models/trained"
DEFAULT_RESULTS_DIR = "results"

# Normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Configuración de detección de rostros
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 6