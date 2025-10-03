"""
Script para probar imágenes nuevas
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.transforms import get_val_transforms
from models.cnn_model import CNNModel
from utils.constants import *


def load_best_model(device):
    """Cargar el mejor modelo entrenado"""
    models_dir = Path("models/trained")
    model_files = list(models_dir.glob("best_model_epoch_*.pth"))
    
    if not model_files:
        raise FileNotFoundError("No se encontraron modelos entrenados")
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    checkpoint = torch.load(latest_model, map_location=device)
    
    model = CNNModel(
        num_classes=NUM_CLASSES,
        input_size=(IMAGE_SIZE, IMAGE_SIZE),
        num_channels=CHANNELS,
        dropout_prob=0.5
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modelo cargado: {latest_model}")
    print(f"Accuracy del modelo: {checkpoint['val_acc']:.4f}")
    
    return model


def preprocess_image(image_path, target_size=(100, 100)):
    """Preprocesar imagen para el modelo"""
    try:
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        
        # Redimensionar
        image = image.resize(target_size)
        
        # Aplicar transformaciones
        transform = get_val_transforms()
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor, image
    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return None, None


def predict_emotion(model, image_tensor, device):
    """Predecir emoción de una imagen"""
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Obtener probabilidades
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]


def plot_prediction_results(image, predicted_emotion, confidence, probabilities, save_path=None):
    """Visualizar resultados de predicción"""
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mostrar imagen
    ax1.imshow(image)
    ax1.set_title(f'Predicción: {emotion_names[predicted_emotion]}\nConfianza: {confidence:.3f}')
    ax1.axis('off')
    
    # Mostrar probabilidades
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotion_names)))
    bars = ax2.bar(emotion_names, probabilities, color=colors)
    ax2.set_title('Probabilidades por Emoción')
    ax2.set_ylabel('Probabilidad')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Resaltar la predicción
    bars[predicted_emotion].set_color('red')
    
    # Añadir valores en las barras
    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Resultado guardado: {save_path}")
    
    plt.show()


def test_single_image(image_path, model, device, save_results=True):
    """Probar una sola imagen"""
    print(f"\nProbando imagen: {image_path}")
    
    # Preprocesar imagen
    image_tensor, original_image = preprocess_image(image_path)
    
    if image_tensor is None:
        return
    
    # Predecir
    predicted_class, confidence, probabilities = predict_emotion(model, image_tensor, device)
    emotion_name = EMOTION_CLASSES[predicted_class]
    
    print(f"Emoción predicha: {emotion_name}")
    print(f"Confianza: {confidence:.4f}")
    
    # Guardar resultados si se solicita
    if save_results:
        results_dir = Path("results/predictions")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        save_path = results_dir / f"{image_name}_prediction.png"
        
        plot_prediction_results(original_image, predicted_class, confidence, 
                              probabilities, save_path)
    else:
        plot_prediction_results(original_image, predicted_class, confidence, probabilities)


def test_directory(images_dir, model, device, max_images=10):
    """Probar múltiples imágenes de un directorio"""
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        print(f"Error: {images_dir} no existe")
        return
    
    # Extensiones de imagen soportadas
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No se encontraron imágenes en {images_dir}")
        return
    
    # Limitar número de imágenes
    image_files = image_files[:max_images]
    
    print(f"Probando {len(image_files)} imágenes...")
    
    results = []
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    
    # Crear directorio para resultados
    results_dir = Path("results/batch_predictions")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for image_file in image_files:
        image_tensor, original_image = preprocess_image(image_file)
        if image_tensor is not None:
            predicted_class, confidence, probabilities = predict_emotion(model, image_tensor, device)
            emotion_name = EMOTION_CLASSES[predicted_class]
            
            result = {
                'file': image_file.name,
                'emotion': emotion_name,
                'confidence': confidence,
                'probabilities': probabilities
            }
            results.append(result)
            
            print(f"{image_file.name}: {emotion_name} ({confidence:.4f})")
            
            # Guardar predicción individual
            save_path = results_dir / f"{image_file.stem}_prediction.png"
            plot_prediction_results(original_image, predicted_class, confidence, 
                                  probabilities, save_path)
    
    # Guardar resumen de resultados
    save_batch_summary(results, results_dir, images_dir.name)
    
    return results


def save_batch_summary(results, results_dir, folder_name):
    """Guardar resumen de predicciones en lote"""
    if not results:
        return
    
    # Crear resumen de texto
    summary_path = results_dir / f"{folder_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"RESUMEN DE PREDICCIONES - {folder_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total de imágenes: {len(results)}\n\n")
        
        # Estadísticas por emoción
        emotion_counts = {}
        confidence_by_emotion = {}
        
        for result in results:
            emotion = result['emotion']
            confidence = result['confidence']
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                confidence_by_emotion[emotion] = []
            
            emotion_counts[emotion] += 1
            confidence_by_emotion[emotion].append(confidence)
        
        f.write("DISTRIBUCIÓN DE PREDICCIONES:\n")
        f.write("-" * 30 + "\n")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(results)) * 100
            avg_confidence = np.mean(confidence_by_emotion[emotion])
            f.write(f"{emotion}: {count} ({percentage:.1f}%) - Confianza promedio: {avg_confidence:.3f}\n")
        
        f.write("\nDETALLE POR IMAGEN:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"{result['file']}: {result['emotion']} ({result['confidence']:.4f})\n")
    
    # Crear gráfico de resumen
    plot_batch_summary(results, results_dir, folder_name)
    
    print(f"Resumen guardado: {summary_path}")


def plot_batch_summary(results, results_dir, folder_name):
    """Crear gráficos de resumen de predicciones en lote"""
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    
    # Contar predicciones por emoción
    emotion_counts = {emotion: 0 for emotion in emotion_names}
    confidences = []
    
    for result in results:
        emotion_counts[result['emotion']] += 1
        confidences.append(result['confidence'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de predicciones
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    bars1 = ax1.bar(emotions, counts, color='skyblue')
    ax1.set_title(f'Distribución de Predicciones - {folder_name}')
    ax1.set_xlabel('Emociones')
    ax1.set_ylabel('Cantidad')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars1, counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
    
    # 2. Distribución de confianza
    ax2.hist(confidences, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('Distribución de Confianza')
    ax2.set_xlabel('Confianza')
    ax2.set_ylabel('Frecuencia')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Media: {np.mean(confidences):.3f}')
    ax2.legend()
    
    # 3. Confianza promedio por emoción
    emotion_confidences = {}
    for result in results:
        emotion = result['emotion']
        if emotion not in emotion_confidences:
            emotion_confidences[emotion] = []
        emotion_confidences[emotion].append(result['confidence'])
    
    avg_confidences = [np.mean(emotion_confidences.get(emotion, [0])) 
                      for emotion in emotions]
    
    bars3 = ax3.bar(emotions, avg_confidences, color='lightcoral')
    ax3.set_title('Confianza Promedio por Emoción')
    ax3.set_xlabel('Emociones')
    ax3.set_ylabel('Confianza Promedio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1)
    
    for bar, conf in zip(bars3, avg_confidences):
        if conf > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
    
    # 4. Top imágenes con mayor confianza
    top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:10]
    top_files = [r['file'][:15] + '...' if len(r['file']) > 15 else r['file'] 
                for r in top_results]
    top_confidences = [r['confidence'] for r in top_results]
    
    bars4 = ax4.barh(range(len(top_files)), top_confidences, color='gold')
    ax4.set_title('Top 10 Predicciones por Confianza')
    ax4.set_xlabel('Confianza')
    ax4.set_yticks(range(len(top_files)))
    ax4.set_yticklabels(top_files)
    ax4.set_xlim(0, 1)
    
    plt.tight_layout()
    
    summary_plot_path = results_dir / f"{folder_name}_summary.png"
    plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Gráfico de resumen guardado: {summary_plot_path}")


def main():
    """Función principal"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    try:
        model = load_best_model(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ejecuta primero: python scripts/train_model.py")
        return
    
    # Opciones de uso
    print("\nOpciones de uso:")
    print("1. Probar una imagen específica")
    print("2. Probar imágenes de un directorio")
    print("3. Usar imágenes de ejemplo del dataset")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == "1":
        image_path = input("Ingresa la ruta de la imagen: ").strip()
        if os.path.exists(image_path):
            test_single_image(image_path, model, device)
        else:
            print(f"Error: {image_path} no existe")
    
    elif choice == "2":
        images_dir = input("Ingresa la ruta del directorio: ").strip()
        max_images = int(input("Número máximo de imágenes a probar (default 10): ") or "10")
        test_directory(images_dir, model, device, max_images)
    
    elif choice == "3":
        # Usar imágenes de ejemplo del dataset de validación
        val_dir = Path("data/processed/val")
        if val_dir.exists():
            emotions = [d.name for d in val_dir.iterdir() if d.is_dir()]
            print(f"Emociones disponibles: {emotions}")
            
            selected_emotion = input("Selecciona una emoción: ").strip()
            if selected_emotion in emotions:
                emotion_dir = val_dir / selected_emotion
                test_directory(emotion_dir, model, device, max_images=5)
            else:
                print("Emoción no válida")
        else:
            print("Dataset de validación no encontrado")
    
    else:
        print("Opción no válida")


if __name__ == "__main__":
    main()
