"""
Constantes utilizadas en todo el proyecto
"""

# Emociones y sus índices
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

# Configuración de imágenes
IMAGE_SIZE = 100
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
