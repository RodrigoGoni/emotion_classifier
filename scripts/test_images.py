"""
Script actualizado para probar imágenes individuales con análisis detallado
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.transforms import get_val_transforms
from src.models.cnn_model import CNNModel
from src.utils.constants import EMOTION_CLASSES, NUM_CLASSES, IMAGE_SIZE, CHANNELS


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


def predict_single_image(model, image_path, transform, device):
    """Predecir emoción para una sola imagen con análisis detallado"""
    try:
        # Cargar imagen original
        original_image = Image.open(image_path).convert('RGB')
        
        # Aplicar transformaciones
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Predecir
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0][predicted_class].item()
            all_scores = probabilities.cpu().numpy()[0]
        
        predicted_emotion = EMOTION_CLASSES[predicted_class]
        
        return {
            'original_image': original_image,
            'processed_tensor': image_tensor,
            'predicted_class': predicted_class,
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'all_scores': all_scores,
            'success': True
        }
        
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return {'success': False, 'error': str(e)}


def create_detailed_plot(result, image_path, save_path=None):
    """Crear plot detallado con imagen original, preprocesada, scores y análisis"""
    if not result['success']:
        print(f"No se puede crear plot: {result['error']}")
        return
    
    # Preparar imagen preprocesada para visualización
    preprocessed_display = result['processed_tensor'].squeeze(0).cpu()
    # Desnormalizar
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    preprocessed_display = preprocessed_display * std.view(3, 1, 1) + mean.view(3, 1, 1)
    preprocessed_display = torch.clamp(preprocessed_display, 0, 1)
    preprocessed_display = preprocessed_display.permute(1, 2, 0).numpy()
    
    # Crear figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Imagen original
    ax1.imshow(result['original_image'])
    ax1.set_title(f'Imagen Original\n{Path(image_path).name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Imagen preprocesada
    ax2.imshow(preprocessed_display)
    ax2.set_title(f'Imagen Preprocesada\n{IMAGE_SIZE}x{IMAGE_SIZE}, Normalizada', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Scores por clase
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    scores = result['all_scores']
    
    colors = ['red' if i == result['predicted_class'] else 'lightblue' for i in range(NUM_CLASSES)]
    bars = ax3.bar(emotion_names, scores, color=colors, alpha=0.7)
    ax3.set_title('Scores por Clase', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probabilidad')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Resaltar clase ganadora
    bars[result['predicted_class']].set_color('red')
    bars[result['predicted_class']].set_alpha(1.0)
    
    # Añadir valores en barras
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Análisis detallado
    ax4.axis('off')
    
    # Top 3 emociones
    top_3_indices = np.argsort(scores)[::-1][:3]
    
    analysis_text = f"""
PREDICCIÓN PRINCIPAL:
    {result['predicted_emotion']}
    Confianza: {result['confidence']:.3f}

CLASE GANADORA:
    Clase {result['predicted_class']}
    Score: {scores[result['predicted_class']]:.3f}

TOP 3 PREDICCIONES:
"""
    
    for i, idx in enumerate(top_3_indices):
        analysis_text += f"    {i+1}. {emotion_names[idx]}: {scores[idx]:.3f}\n"
    
    analysis_text += f"""
INFORMACIÓN TÉCNICA:
    Modelo: CNNModel
    Input: {IMAGE_SIZE}x{IMAGE_SIZE}x{CHANNELS}
    Clases: {NUM_CLASSES}
    Transformaciones: Resize + Normalización
"""
    
    ax4.text(0.1, 0.9, analysis_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Título principal
    main_title = f'Predicción: {result["predicted_emotion"]} (Confianza: {result["confidence"]:.3f})'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_single_image_interactive():
    """Probar una imagen individual de forma interactiva"""
    print("\n=== PREDICCIÓN DE IMAGEN INDIVIDUAL ===")
    
    image_path = input("Ingresa la ruta de la imagen: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: La imagen {image_path} no existe")
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(device)
    transform = get_val_transforms()
    
    # Predecir
    result = predict_single_image(model, image_path, transform, device)
    
    if not result['success']:
        return
    
    # Mostrar resultados
    print(f"\nRESULTADOS:")
    print(f"Emoción predicha: {result['predicted_emotion']}")
    print(f"Confianza: {result['confidence']:.4f}")
    print(f"Clase: {result['predicted_class']}")
    
    # Mostrar top 3
    print(f"\nTop 3 predicciones:")
    top_3_indices = np.argsort(result['all_scores'])[::-1][:3]
    for i, idx in enumerate(top_3_indices):
        emotion = EMOTION_CLASSES[idx]
        score = result['all_scores'][idx]
        print(f"  {i+1}. {emotion}: {score:.4f}")
    
    # Preguntar si guardar plot
    save_option = input("\n¿Guardar plot detallado? (s/n): ").strip().lower()
    if save_option == 's':
        results_dir = Path("results/single_predictions")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{Path(image_path).stem}_{result['predicted_emotion']}.png"
        save_path = results_dir / filename
        
        create_detailed_plot(result, image_path, save_path)
    else:
        create_detailed_plot(result, image_path)


def test_directory_images():
    """Probar todas las imágenes de un directorio"""
    print("\n=== PREDICCIÓN DE DIRECTORIO ===")
    
    directory_path = input("Ingresa la ruta del directorio: ").strip()
    
    if not os.path.exists(directory_path):
        print(f"Error: El directorio {directory_path} no existe")
        return
    
    # Buscar imágenes
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(directory_path).glob(f"*{ext}"))
        image_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("No se encontraron imágenes en el directorio")
        return
    
    print(f"Encontradas {len(image_files)} imágenes")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(device)
    transform = get_val_transforms()
    
    # Crear directorio de resultados
    results_dir = Path("results/batch_predictions")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar imágenes
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcesando {i}/{len(image_files)}: {image_path.name}")
        
        result = predict_single_image(model, image_path, transform, device)
        
        if result['success']:
            print(f"  Predicción: {result['predicted_emotion']} (confianza: {result['confidence']:.3f})")
            
            # Guardar plot individual
            filename = f"{image_path.stem}_{result['predicted_emotion']}.png"
            save_path = results_dir / filename
            create_detailed_plot(result, image_path, save_path)
            
            results.append({
                'file': image_path.name,
                'emotion': result['predicted_emotion'],
                'confidence': result['confidence'],
                'class': result['predicted_class']
            })
    
    # Crear resumen
    if results:
        summary_path = results_dir / "batch_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RESUMEN DE PREDICCIONES EN LOTE\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total de imágenes procesadas: {len(results)}\n\n")
            
            # Contar por emoción
            emotion_counts = {}
            for result in results:
                emotion = result['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            f.write("DISTRIBUCIÓN DE EMOCIONES:\n")
            f.write("-" * 25 + "\n")
            for emotion, count in sorted(emotion_counts.items()):
                percentage = (count / len(results)) * 100
                f.write(f"{emotion}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nDETALLE POR IMAGEN:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                f.write(f"{result['file']}: {result['emotion']} ({result['confidence']:.3f})\n")
        
        print(f"\nResumen guardado: {summary_path}")
        print(f"Plots individuales en: {results_dir}")


def show_sample_dataset_images():
    """Mostrar imágenes de ejemplo del dataset de entrenamiento"""
    print("\n=== IMÁGENES DE EJEMPLO DEL DATASET ===")
    
    dataset_path = Path("data/processed/train")
    if not dataset_path.exists():
        print("Dataset de entrenamiento no encontrado")
        print("Ejecuta primero: python scripts/create_balanced_dataset.py")
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_best_model(device)
    transform = get_val_transforms()
    
    # Crear directorio de resultados
    results_dir = Path("results/dataset_samples")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Tomar una imagen de cada emoción
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    
    for emotion in emotion_names:
        emotion_dir = dataset_path / emotion
        if emotion_dir.exists():
            image_files = list(emotion_dir.glob("*.jpg"))
            if image_files:
                # Tomar la primera imagen
                image_path = image_files[0]
                print(f"\nProcesando ejemplo de {emotion}: {image_path.name}")
                
                result = predict_single_image(model, image_path, transform, device)
                
                if result['success']:
                    print(f"  Predicción: {result['predicted_emotion']} (confianza: {result['confidence']:.3f})")
                    
                    # Crear plot
                    filename = f"sample_{emotion}_{result['predicted_emotion']}.png"
                    save_path = results_dir / filename
                    create_detailed_plot(result, image_path, save_path)
    
    print(f"\nEjemplos guardados en: {results_dir}")


def main():
    """Función principal con menú interactivo"""
    print("=== PREDICTOR DE EMOCIONES ACTUALIZADO ===")
    
    # Verificar que hay modelos entrenados
    models_dir = Path("models/trained")
    if not models_dir.exists() or not list(models_dir.glob("*.pth")):
        print("Error: No se encontraron modelos entrenados")
        print("Ejecuta primero: python scripts/train_model.py")
        return
    
    while True:
        print("\nOpciones disponibles:")
        print("1. Predecir imagen individual")
        print("2. Predecir imágenes de un directorio")
        print("3. Mostrar ejemplos del dataset de entrenamiento")
        print("4. Salir")
        
        try:
            option = input("\nSelecciona una opción (1-4): ").strip()
            
            if option == '1':
                test_single_image_interactive()
            elif option == '2':
                test_directory_images()
            elif option == '3':
                show_sample_dataset_images()
            elif option == '4':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()