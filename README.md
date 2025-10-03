# Clasificador de Emociones con CNN

Clasificador de emociones usando CNN con PyTorch.

## Instalación

```bash
pip install -r requirements.txt
```

## Dataset

Descargar desde: https://drive.google.com/file/d/1auZ64-CEfa4tx16cVq9TdibsdKwQY9jN/view?usp=sharing

```bash
python scripts/download_data.py
```

## Uso

### Entrenamiento
```bash
python scripts/train_model.py
```

### Evaluación
```bash
python scripts/evaluate_model.py
```

### Predicción
```bash
python scripts/test_images.py --image_path image.jpg
python scripts/test_images.py --image_path image.jpg --use_face_detection
```

## Estructura

```
src/
├── data/          # Dataset y transformaciones
├── models/        # Arquitectura CNN
├── inference/     # Predicciones y detección de rostros
└── utils/         # Utilidades

config/            # Configuración
data/              # Datasets
models/trained/    # Modelos guardados
results/           # Resultados
notebooks/         # Jupyter notebooks
scripts/           # Scripts ejecutables
```

## Emociones

angry, disgust, fear, happy, neutral, sad, surprise
