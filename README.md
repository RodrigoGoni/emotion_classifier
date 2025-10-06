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
Respuestas
1 las tranformaciones se hiceron con la version dos de la libreria de torchvision, primero hay que tener mucho cuidado con el tipo de trasformaciones que se ahcen en este caso porque al tener que clasificar emociones no se puede ser muy drastico con transformaciones que modifique la dorma o la posicion de la cara por que esto puede modificar la expresion en si misma.
Despues se eligio unas transformaciones suabes en cuento a movimiento de la imagen y un poco mas intensar (aumentando la probabilidad p de qeu se aplique) en las que blurean o agregan ruido a la imagen
Lo que se puede obserbar es que hay un desvalance demasiado grande en el dataset esto siempre degrada la performance del modelo y no solo en las clases que tienen pocas cantidad de daros si no tambien en las que estan sobre muestreadas, entre la clase que mastiene felicidad y la que menos tiene miedo hay apoximadamente un *20, esto en la evalucacion con los datos de calidacion del datast no se ve reflejado en los resultado pero descargue un dataset mucho mas rico y grande para testear que se llama affecnet y hay clara evidencia del desvalanceo en la primeras iteraciones de entrenamiento tenia un resultado de de 85 en felicidad y 20 en miedo 
2.
LA eleccion del modelo queira que fuese sencillo para entenerlo yo y poder trasmitirlo a ustede en un oirmer momento pense en usar una arquitectura resnet o recidual network pero creo que iba a compejisar mucho el problema y no estaba 100seguro si iba aportar una mejoria realmente notable lo que seguramente aprotaria una mejora notable en este tipod de problemas es capas de atencion ya que necesitamos detectar caracteriticas de la cara y para ello debermos prestarle atencion a ciertas partes del rostro, de nuevo esto compejisa mucho el tp, y no aunque yo en mi maquina puedo entrenar la red no estoy seguro si podria correr este modelo en google colab.
Bloques Convolucionales
Por lo que decidi basarme en lo que vimos en clase y utiliza una arquitectura tipica de CNN, con algunas caracteristicas sutiles pero importantes para esta tarea fui aumentando la cantidad de filtro desde el primcipo al final de la red (32 → 64 → 128 → 256) por que me parece que tenemos que encontrar carcteristicas mas puntuales para esta tarea, las capas finales de la red aprenden caracteristicas puntuales mientras las primeras caracteriscticas generales 
La dimension de los kernels tambien tiene detalles que por ahi se pasan use 3*3 esto tiene justificacion en lo que se llama campo reseptivo efectivo de la red 
estp es algo como lo que voy a explicar por ejemplo si me hubiera decidido por kernels de 5*5 si aplcio una capa de conv con ese kernel tiene el mismo efecto que aplicar dos veces un kernel de 3*3 pero tiene la desventaja que con una sola aplicacion pierdo la posibilida de tener dos capas de no linealidad osea capturo menos carcateristicas no lineales y tiene mas parametro 25 el kernel de 5*5 y 18 en los dos kernels de 3*3 y bueno siempre un kernel mas chico vas a poder ver carateristicas ricas y mas contexto de la imagen
en el pading no hay nado demaciado especial muy standar
el max pol solo para reducir la complejidad comp (reduce en la mitad la dimencion original) y tambien la hace robusta para pequenas traslaciones de las caracteriticas 
Bloque Clasificador (backbone)
aqui estan los famosos backbine de las redes CNN 
simple una fulluconected de 512 neuronas conectados a la salida de 7 neurona
le agrege un dropout para evitar el overfiting 
esta parte es la encargada una vez extraida las caracteristicas en clasificarlas en las emociones de salida 
en cuanto a las no linealidades nada muy extraordinario use ReLU simplemente es mas eficiente y lo mas estandar, podria aclarar que es un poco mas eficiente para entrenar las capas de entrada porque su derivada no sufre de poblemas de saturacion en los extremos pero no cero que sea tan importante. 
Otra cosa tal vez importante es la capa de salida que es una nn.CrossEntropyLoss implisitamente una Softmax que basicamente hace que la suma de las probavilidades de la salida de 1 es lo estandar en clasificacion multi clase 
Bueno para las demas parte del entrenamiento no creo que valga la pena aclarar mas cosa por que utilice cosas muy standars como un algoritmo adam con un lr de 1e-3 que es lo recomendado 
LA cantidad de capas oculyas 4 me parecieron bien y estandar 
4. el test que propuse fue samplear 100 casos del dataset affectnet y matchearlo con nuestaras clases para poder validar la performance del modelo . desarrolle un test para esto y evaluar las metricas propuestas por el tp
Como conclucion un poco las mensione en 1 el dataset affecnet prensta muy mala performace en la calses desvalnceadas como solucion yo propuse una aumentacion dle dataset y un balanceo , para el dataset que propuso la catedra considero que el modelo anda satisfactoriamente 8/10 podria mejorarce con las tecnicas que propuse en el punto 1 un extarctor de caracteristicas mas complejo con cpas de atenccion y un clasificador mejor con atencion tambien podria ser (la cantidad de hiperparametros crece muchismo)
