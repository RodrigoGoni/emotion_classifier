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

1. Transformaciones y Desbalanceo del Dataset
Las transformaciones se realizaron con la versión dos de la librería torchvision. En este caso, se debe tener especial cuidado con el tipo de transformaciones aplicadas, ya que al clasificar emociones, no se pueden utilizar modificaciones drásticas que alteren la forma o posición de la cara, puesto que esto podría cambiar la expresión en sí misma.

Por ello, se optó por transformaciones suaves en cuanto al movimiento de la imagen y algunas más intensas (aumentando la probabilidad p de su aplicación) enfocadas en desenfoque (blur) o adición de ruido.

Se observó un desbalanceo significativo en el dataset, lo cual siempre degrada el rendimiento del modelo, afectando no solo a las clases con pocos datos, sino también a las sobremuestreadas. Existe una proporción aproximada de 20 a 1 entre la clase más frecuente (felicidad) y la menos frecuente (miedo). Aunque esto no se vio reflejado en los resultados con los datos de validación del dataset original, al evaluar con un dataset más grande y rico (AffecNet), la evidencia del desbalanceo fue clara. En las primeras iteraciones de entrenamiento con AffecNet, se obtuvo un resultado de 85% en felicidad y 20% en miedo.

2. Elección y Arquitectura del Modelo
La elección del modelo se basó en la sencillez para su comprensión y posterior explicación. Inicialmente, se consideró una arquitectura ResNet (Residual Network), pero se pensó que complejizaría demasiado el problema sin garantizar una mejora notable. Una mejora significativa para este tipo de tareas sería la inclusión de Capas de Atención (Attention Layers), ya que se requiere detectar características faciales específicas, prestando atención a ciertas partes del rostro. No obstante, esto incrementaba la complejidad del trabajo y había dudas sobre si podría ejecutarse eficientemente en Google Colab.

Bloques Convolucionales (CNN)
Se decidió basar la arquitectura en una Red Neuronal Convolucional (CNN) típica, siguiendo lo visto en clase, con algunas características sutiles pero importantes:

Aumento de Filtros: Se incrementó la cantidad de filtros desde el principio hasta el final de la red (32→64→128→256). Esto se debe a que las capas finales aprenden características puntuales, mientras que las primeras aprenden características generales, y para la tarea de clasificación de emociones, se necesitan características más específicas.

Tamaño del Kernel: Se usaron kernels de 3×3 basándose en el concepto de Campo Receptivo Efectivo. Aplicar un solo kernel de 5×5 tiene un efecto similar al de aplicar dos veces un kernel de 3×3. Sin embargo, el kernel más pequeño tiene la ventaja de permitir dos capas de no linealidad, capturando más características no lineales, además de tener menos parámetros (18 en los dos kernels de 3×3 frente a 25 en el de 5×5). Un kernel más pequeño puede capturar características ricas y más contexto de la imagen.

Padding y Max Pooling: El padding es estándar. Se utiliza Max Pooling únicamente para reducir la complejidad computacional (reduce la dimensión original a la mitad) y para hacer la red más robusta a pequeñas traslaciones de las características.

Bloque Clasificador (Backbone)
Esta parte utiliza el backbone de las redes CNN:

Una capa Totalmente Conectada (Fully Connected) de 512 neuronas, conectada a la capa de salida de 7 neuronas (correspondiente al número de emociones).

Se agregó Dropout para prevenir el sobreajuste (overfitting).

Esta sección es la encargada de clasificar las características extraídas en las emociones de salida.

Funciones de Activación y Pérdida
No Linealidades: Se usó ReLU por su eficiencia y ser la opción más estándar. Es más eficiente para entrenar las capas de entrada, ya que su derivada no sufre problemas de saturación en los extremos.

Capa de Salida: Se utilizó nn.CrossEntropyLoss, que implícitamente incluye una Softmax. Esto es el estándar en clasificación multiclase y asegura que la suma de las probabilidades de salida sea 1.

Entrenamiento
Para el resto de las partes del entrenamiento, se usaron configuraciones estándar, como el algoritmo Adam con una tasa de aprendizaje (lr) de 1e-3. Se emplearon cuatro capas ocultas, consideradas una cantidad adecuada y estándar.

4. Prueba y Conclusiones
La prueba propuesta consistió en muestrear 100 casos del dataset AffecNet y mapearlos con nuestras clases para validar el rendimiento del modelo, evaluando las métricas requeridas.

Conclusión: Como se mencionó en el punto 1, el dataset AffecNet mostró un rendimiento deficiente en las clases desbalanceadas. Como solución, se propone un aumento y balanceo del dataset. Para el dataset provisto por la cátedra, se considera que el modelo tiene un rendimiento satisfactorio (8/10), aunque podría mejorarse con las técnicas antes mencionadas: un extractor de características más complejo, posiblemente con capas de atención, y un clasificador mejorado que también incorpore atención, a pesar de que esto incrementaría considerablemente la cantidad de hiperparámetros.