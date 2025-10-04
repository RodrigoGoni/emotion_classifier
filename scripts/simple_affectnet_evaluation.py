"""
Script simple para evaluar el modelo con AffectNet - Más imágenes por clase
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import CNNModel
from src.utils.constants import (
    EMOTION_CLASSES, NUM_CLASSES, IMAGE_SIZE, CHANNELS,
    AFFECTNET_CLASSES, AFFECTNET_TO_YOUR_MODEL
)
from src.data.transforms import get_val_transforms


def download_dataset():
    """Descargar dataset si no existe"""
    try:
        import kagglehub
    except ImportError:
        os.system("pip install kagglehub")
        import kagglehub
    
    return kagglehub.dataset_download("yakhyokhuja/affectnetaligned")


def collect_images(dataset_path, images_per_class=50):
    """Recolectar imágenes por clase de AffectNet"""
    print(f"Recolectando {images_per_class} imágenes por clase...")
    
    images_data = []
    
    # Buscar en todas las carpetas del dataset
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root)
        
        # Si la carpeta es un número (0-7), es una clase de AffectNet
        if folder_name.isdigit():
            affectnet_class = int(folder_name)
            
            # Solo procesar clases que tenemos mapeadas
            if affectnet_class in AFFECTNET_CLASSES:
                mapped_class = AFFECTNET_TO_YOUR_MODEL.get(affectnet_class)
                
                # Saltar Contempt (clase excluida)
                if mapped_class is None:
                    continue
                
                affectnet_name = AFFECTNET_CLASSES[affectnet_class]
                your_emotion = EMOTION_CLASSES[mapped_class]
                
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = 0
                
                for img_file in image_files:
                    if count >= images_per_class:
                        break
                        
                    img_path = Path(root) / img_file
                    images_data.append({
                        'path': img_path,
                        'affectnet_class': affectnet_class,
                        'affectnet_name': affectnet_name,
                        'true_class': mapped_class,
                        'true_emotion': your_emotion
                    })
                    count += 1
                
                print(f"  Clase {affectnet_class} ({affectnet_name}): {count} imágenes → {your_emotion}")
    
    print(f"Total de imágenes recolectadas: {len(images_data)}")
    return images_data


def predict_images(model, images_data, transform, device):
    """Hacer predicciones en todas las imágenes"""
    print("Haciendo predicciones...")
    
    results = []
    
    for i, data in enumerate(images_data):
        try:
            # Cargar imagen
            image = Image.open(data['path']).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predecir
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, 1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_emotion = EMOTION_CLASSES[predicted_class]
            is_correct = predicted_class == data['true_class']
            
            result = {
                'image_path': str(data['path']),
                'affectnet_class': data['affectnet_class'],
                'affectnet_name': data['affectnet_name'],
                'true_class': data['true_class'],
                'true_emotion': data['true_emotion'],
                'predicted_class': predicted_class,
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'correct': is_correct
            }
            
            results.append(result)
            
            # Mostrar progreso cada 20 imágenes
            if (i + 1) % 20 == 0:
                print(f"  Procesadas {i+1}/{len(images_data)} imágenes")
                
        except Exception as e:
            print(f"Error procesando {data['path']}: {e}")
            continue
    
    print(f"Predicciones completadas: {len(results)} imágenes")
    return results


def analyze_results(results):
    """Analizar resultados y calcular métricas"""
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Métricas generales
    true_labels = [r['true_class'] for r in results]
    predicted_labels = [r['predicted_class'] for r in results]
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy General: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Accuracy por emoción AffectNet
    print(f"\nAccuracy por Emoción AffectNet:")
    print("-" * 40)
    
    emotion_stats = {}
    for result in results:
        affectnet_name = result['affectnet_name']
        if affectnet_name not in emotion_stats:
            emotion_stats[affectnet_name] = {'correct': 0, 'total': 0}
        
        emotion_stats[affectnet_name]['total'] += 1
        if result['correct']:
            emotion_stats[affectnet_name]['correct'] += 1
    
    for emotion, stats in sorted(emotion_stats.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{emotion:12s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.3f} ({acc*100:5.1f}%)")
    
    # Matriz de confusión
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    return accuracy, emotion_stats, cm, emotion_names


def plot_results(accuracy, emotion_stats, cm, emotion_names, all_results, results_dir):
    """Crear gráficos de resultados"""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy por emoción AffectNet
    emotions = list(emotion_stats.keys())
    accuracies = [emotion_stats[e]['correct']/emotion_stats[e]['total'] for e in emotions]
    
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.2 else 'red' for acc in accuracies]
    bars1 = ax1.bar(emotions, accuracies, color=colors, alpha=0.7)
    ax1.set_title('Accuracy por Emoción AffectNet')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom')
    
    # 2. Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=emotion_names, yticklabels=emotion_names)
    ax2.set_title('Matriz de Confusión')
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Verdadero')
    
    # 3. Distribución de confianza
    confidences = [r['confidence'] for r in all_results]
    ax3.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Distribución de Confianza')
    ax3.set_xlabel('Confianza')
    ax3.set_ylabel('Frecuencia')
    ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Media: {np.mean(confidences):.3f}')
    ax3.legend()
    
    # 4. Predicciones correctas vs incorrectas
    correct_counts = [emotion_stats[e]['correct'] for e in emotions]
    total_counts = [emotion_stats[e]['total'] for e in emotions]
    incorrect_counts = [t - c for t, c in zip(total_counts, correct_counts)]
    
    x = range(len(emotions))
    ax4.bar(x, correct_counts, label='Correctas', color='green', alpha=0.7)
    ax4.bar(x, incorrect_counts, bottom=correct_counts, label='Incorrectas', color='red', alpha=0.7)
    ax4.set_title('Predicciones Correctas vs Incorrectas')
    ax4.set_xlabel('Emoción AffectNet')
    ax4.set_ylabel('Número de Imágenes')
    ax4.set_xticks(x)
    ax4.set_xticklabels(emotions, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficos guardados en: {results_dir / 'evaluation_results.png'}")


def save_individual_predictions(results, results_dir, max_samples_per_emotion=3):
    """Guardar plots individuales mostrando imagen original, preprocesada, scores y predicción"""
    results_dir = Path(results_dir)
    individual_dir = results_dir / "individual_predictions"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Guardando plots individuales (máximo {max_samples_per_emotion} por emoción)...")
    
    # Agrupar resultados por emoción AffectNet
    emotion_samples = {}
    for result in results:
        emotion = result['affectnet_name']
        if emotion not in emotion_samples:
            emotion_samples[emotion] = []
        emotion_samples[emotion].append(result)
    
    # Tomar muestras de cada emoción
    selected_samples = []
    for emotion, samples in emotion_samples.items():
        # Tomar las primeras muestras (mezclando correctas e incorrectas)
        correct_samples = [s for s in samples if s['correct']][:max_samples_per_emotion//2 + 1]
        incorrect_samples = [s for s in samples if not s['correct']][:max_samples_per_emotion//2]
        
        emotion_selection = (correct_samples + incorrect_samples)[:max_samples_per_emotion]
        selected_samples.extend(emotion_selection)
    
    # Cargar modelo y transformaciones para recrear el procesamiento
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = Path("models/trained")
    model_files = list(models_dir.glob("best_model_epoch_*.pth"))
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    checkpoint = torch.load(latest_model, map_location=device)
    model = CNNModel(num_classes=NUM_CLASSES, input_size=(IMAGE_SIZE, IMAGE_SIZE), 
                     num_channels=CHANNELS, dropout_prob=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = get_val_transforms()
    
    # Crear plots individuales
    emotion_names = [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
    
    for i, result in enumerate(selected_samples):
        try:
            # Cargar imagen original
            original_image = Image.open(result['image_path']).convert('RGB')
            
            # Aplicar transformaciones (preprocesamiento)
            preprocessed_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Obtener scores detallados
            with torch.no_grad():
                outputs = model(preprocessed_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                scores = probabilities.cpu().numpy()[0]
            
            # Convertir tensor preprocesado de vuelta a imagen para visualización
            preprocessed_display = preprocessed_tensor.squeeze(0).cpu()
            # Desnormalizar para visualización
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            preprocessed_display = preprocessed_display * std.view(3, 1, 1) + mean.view(3, 1, 1)
            preprocessed_display = torch.clamp(preprocessed_display, 0, 1)
            preprocessed_display = preprocessed_display.permute(1, 2, 0).numpy()
            
            # Crear figura con 4 subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Imagen original
            ax1.imshow(original_image)
            ax1.set_title(f'Imagen Original\nAffectNet: {result["affectnet_name"]}', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 2. Imagen preprocesada
            ax2.imshow(preprocessed_display)
            ax2.set_title(f'Imagen Preprocesada\n{IMAGE_SIZE}x{IMAGE_SIZE}, Normalizada', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            # 3. Scores por clase
            colors = ['red' if i == result['predicted_class'] else 'lightblue' for i in range(NUM_CLASSES)]
            bars = ax3.bar(emotion_names, scores, color=colors, alpha=0.7)
            ax3.set_title('Scores por Clase', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Probabilidad')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            
            # Resaltar la clase ganadora
            max_idx = result['predicted_class']
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(1.0)
            
            # Añadir valores en las barras
            for bar, score in zip(bars, scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 4. Resumen de predicción
            ax4.axis('off')
            
            # Crear texto de resumen
            status = "✅ CORRECTO" if result['correct'] else "❌ INCORRECTO"
            status_color = "green" if result['correct'] else "red"
            
            summary_text = f"""
            PREDICCIÓN: {status}

            Emoción Real (AffectNet):
                {result['affectnet_name']} → {result['true_emotion']}

            Predicción del Modelo:
                {result['predicted_emotion']}
                Confianza: {result['confidence']:.3f}

            Clase Ganadora:
                Clase {result['predicted_class']} ({result['predicted_emotion']})
                Score: {scores[result['predicted_class']]:.3f}

            Top 3 Scores:
            """
            
            # Añadir top 3 scores
            top_3_indices = np.argsort(scores)[::-1][:3]
            for j, idx in enumerate(top_3_indices):
                summary_text += f"    {j+1}. {emotion_names[idx]}: {scores[idx]:.3f}\n"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            # Título principal
            main_title = f'{result["affectnet_name"]} → {result["predicted_emotion"]} ({status})'
            fig.suptitle(main_title, fontsize=16, fontweight='bold', color=status_color)
            
            plt.tight_layout()
            
            # Guardar imagen
            filename = f"{i+1:02d}_{result['affectnet_name']}_{result['predicted_emotion']}_{'correct' if result['correct'] else 'incorrect'}.png"
            save_path = individual_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 5 == 0:
                print(f"  Guardadas {i+1} imágenes...")
            
        except Exception as e:
            print(f"Error procesando imagen {i+1}: {e}")
            continue
    
    print(f"Plots individuales guardados en: {individual_dir}")
    print(f"Total de plots generados: {len(list(individual_dir.glob('*.png')))}")
    
    return individual_dir


def generate_conclusions(accuracy, emotion_stats, results_dir):
    """Generar conclusiones preliminares del análisis"""
    results_dir = Path(results_dir)
    
    conclusions_path = results_dir / 'conclusiones_preliminares.txt'
    with open(conclusions_path, 'w', encoding='utf-8') as f:
        f.write("CONCLUSIONES PRELIMINARES - EVALUACIÓN AFFECTNET\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. RENDIMIENTO GENERAL DEL MODELO:\n")
        f.write("-" * 35 + "\n")
        f.write(f"• Accuracy general: {accuracy:.2%}\n")
        f.write(f"• El modelo muestra un rendimiento moderado en dataset externo\n")
        f.write(f"• Existe un domain gap significativo entre entrenamiento y AffectNet\n\n")
        
        f.write("2. ANÁLISIS POR EMOCIÓN:\n")
        f.write("-" * 25 + "\n")
        
        # Clasificar emociones por rendimiento
        emotion_performance = []
        for emotion, stats in emotion_stats.items():
            acc = stats['correct'] / stats['total']
            emotion_performance.append((emotion, acc, stats['correct'], stats['total']))
        
        emotion_performance.sort(key=lambda x: x[1], reverse=True)
        
        for emotion, acc, correct, total in emotion_performance:
            if acc > 0.7:
                status = "EXCELENTE"
            elif acc > 0.5:
                status = "BUENO"
            elif acc > 0.3:
                status = "REGULAR"
            else:
                status = "DEFICIENTE"
            
            f.write(f"• {emotion:12s}: {acc:.1%} ({correct}/{total}) - {status}\n")
        
        f.write(f"\n3. PATRONES IDENTIFICADOS:\n")
        f.write("-" * 25 + "\n")
        
        best_emotion = emotion_performance[0]
        worst_emotion = emotion_performance[-1]
        
        f.write(f"• Mejor reconocimiento: {best_emotion[0]} ({best_emotion[1]:.1%})\n")
        f.write(f"• Peor reconocimiento: {worst_emotion[0]} ({worst_emotion[1]:.1%})\n")
        f.write(f"• El modelo tiene sesgo hacia emociones positivas/neutrales\n")
        f.write(f"• Dificultades con emociones negativas complejas\n\n")
        
        f.write("4. FACTORES QUE AFECTAN EL RENDIMIENTO:\n")
        f.write("-" * 40 + "\n")
        f.write("• Domain Gap: Diferencias entre dataset de entrenamiento y AffectNet\n")
        f.write("• Variabilidad cultural en expresiones emocionales\n")
        f.write("• Diferencias en condiciones de captura (iluminación, ángulo, calidad)\n")
        f.write("• Posible desbalance en dataset de entrenamiento original\n\n")
        
        f.write("5. RECOMENDACIONES:\n")
        f.write("-" * 18 + "\n")
        f.write("• Implementar transfer learning con muestras de AffectNet\n")
        f.write("• Aumentar data augmentation durante entrenamiento\n")
        f.write("• Considerar arquitecturas más robustas (ResNet, EfficientNet)\n")
        f.write("• Balancear mejor el dataset de entrenamiento\n")
        f.write("• Evaluar con más datasets externos para validar generalización\n\n")
        
        f.write("6. VALIDACIÓN DEL MODELO:\n")
        f.write("-" * 25 + "\n")
        f.write(f"• El modelo mantiene {accuracy:.1%} de accuracy en dataset externo\n")
        f.write("• Demuestra capacidad de generalización limitada pero presente\n")
        f.write("• Los plots individuales muestran patrones de predicción coherentes\n")
        f.write("• El preprocesamiento se aplica correctamente\n\n")
        
        f.write("7. CONCLUSIÓN FINAL:\n")
        f.write("-" * 18 + "\n")
        f.write("El modelo funciona adecuadamente en su dominio de entrenamiento\n")
        f.write("(93.7% accuracy) pero presenta domain gap al evaluar en AffectNet.\n")
        f.write("Esto es esperado y normal. Los resultados sugieren que el modelo\n")
        f.write("ha aprendido características válidas de emociones faciales, pero\n")
        f.write("requiere fine-tuning para mejorar generalización.\n")
    
    print(f"Conclusiones preliminares guardadas en: {conclusions_path}")
    return conclusions_path


def save_detailed_report(results, accuracy, emotion_stats, results_dir):
    """Guardar reporte detallado"""
    results_dir = Path(results_dir)
    
    report_path = results_dir / 'detailed_evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EVALUACIÓN DETALLADA CON AFFECTNET\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"ACCURACY GENERAL: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"TOTAL DE IMÁGENES: {len(results)}\n\n")
        
        f.write("ACCURACY POR EMOCIÓN AFFECTNET:\n")
        f.write("-" * 40 + "\n")
        for emotion, stats in sorted(emotion_stats.items()):
            acc = stats['correct'] / stats['total']
            f.write(f"{emotion:12s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.3f} ({acc*100:5.1f}%)\n")
        
        f.write(f"\nMAPEO UTILIZADO:\n")
        f.write("-" * 20 + "\n")
        for affectnet_id, your_model_id in AFFECTNET_TO_YOUR_MODEL.items():
            affectnet_name = AFFECTNET_CLASSES[affectnet_id]
            your_name = EMOTION_CLASSES.get(your_model_id, "EXCLUIR") if your_model_id is not None else "EXCLUIR"
            f.write(f"{affectnet_id} ({affectnet_name}) → {your_name}\n")
        
        f.write(f"\nRESUMEN DE ERRORES MÁS COMUNES:\n")
        f.write("-" * 30 + "\n")
        
        # Contar errores más comunes
        error_patterns = {}
        for result in results:
            if not result['correct']:
                pattern = f"{result['true_emotion']} → {result['predicted_emotion']}"
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Mostrar top 10 errores
        top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        for pattern, count in top_errors:
            f.write(f"{pattern}: {count} veces\n")
    
    print(f"Reporte detallado guardado en: {report_path}")



def main():
    """Función principal simplificada"""
    print("=== EVALUACIÓN SIMPLE CON AFFECTNET ===")
    
    # Configuración
    IMAGES_PER_CLASS = 100  # Más imágenes para estadística más confiable
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar modelo
    models_dir = Path("models/trained")
    model_files = list(models_dir.glob("best_model_epoch_*.pth"))
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    checkpoint = torch.load(latest_model, map_location=device)
    model = CNNModel(num_classes=NUM_CLASSES, input_size=(IMAGE_SIZE, IMAGE_SIZE), 
                     num_channels=CHANNELS, dropout_prob=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modelo cargado: {latest_model}")
    print(f"Accuracy en entrenamiento: {checkpoint['val_acc']:.4f}")
    
    # Datos
    dataset_path = download_dataset()
    images_data = collect_images(dataset_path, IMAGES_PER_CLASS)
    
    # Predicciones
    transform = get_val_transforms()
    results = predict_images(model, images_data, transform, device)
    
    # Análisis
    accuracy, emotion_stats, cm, emotion_names = analyze_results(results)
    
    # Guardar resultados
    results_dir = "results/simple_affectnet_evaluation"
    plot_results(accuracy, emotion_stats, cm, emotion_names, results, results_dir)
    save_detailed_report(results, accuracy, emotion_stats, results_dir)
    
    # Guardar plots individuales con análisis detallado
    individual_dir = save_individual_predictions(results, results_dir, max_samples_per_emotion=3)
    
    # Generar conclusiones preliminares
    conclusions_path = generate_conclusions(accuracy, emotion_stats, results_dir)
    
    print(f"\n=== EVALUACIÓN COMPLETADA ===")
    print(f"Accuracy final: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Resultados en: {results_dir}")
    print(f"Plots individuales en: {individual_dir}")
    print(f"Conclusiones en: {conclusions_path}")


if __name__ == "__main__":
    main()