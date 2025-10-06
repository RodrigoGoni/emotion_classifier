"""Script para evaluar el modelo entrenado con gestión de versiones"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pathlib import Path
from PIL import Image

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import EmotionDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluate import Evaluator
from src.utils.constants import EMOTION_LABELS, NUM_CLASSES
from src.utils.config_manager import ConfigManager
from src.utils.test_dataset import load_test_affectnet_dataset, verify_test_dataset


def convert_numpy_types(obj):
    """Convierte recursivamente tipos numpy a tipos Python nativos para JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def detect_face(image_path, face_cascade):
    """Detecta y extrae rostro de una imagen"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        center_x, center_y = x + w // 2, y + h // 2
        side = max(w, h)
        half_side = side // 2
        
        x1 = max(center_x - half_side, 0)
        y1 = max(center_y - half_side, 0)
        x2 = min(center_x + half_side, image.shape[1])
        y2 = min(center_y + half_side, image.shape[0])
        
        cropped_face = image[y1:y2, x1:x2]
        return cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    
    return None


def predict_image(model, image_array, transform, device):
    """Predice emoción de una imagen"""
    if image_array is None:
        return None
    
    pil_image = Image.fromarray(image_array)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        scores = probabilities.cpu().numpy()[0]
        predicted_class = np.argmax(scores)
        confidence = scores[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'predicted_emotion': EMOTION_LABELS[predicted_class],
        'confidence': confidence
    }


def plot_affectnet_face_detection_analysis(results, model, transform, device, results_dir):
    """Análisis con detección de rostros"""
    
    selected_images = []
    emotions_seen = set()
    
    # Buscar una imagen por emoción, priorizando diversidad
    for result in results:
        true_emotion = result['true_emotion']
        if (true_emotion not in emotions_seen and 
            result['original_pred']):
            selected_images.append(result)
            emotions_seen.add(true_emotion)
            if len(selected_images) >= 6:
                break
    
    for result in results:
        if len(selected_images) >= 6:
            break
        if result not in selected_images and result['original_pred']:
            selected_images.append(result)
    
    # Inicializar detector de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Crear figura para análisis completo (6 filas x 4 columnas)
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    
    face_detection_results = []
    
    for idx, result in enumerate(selected_images[:6]):
        # Cargar imagen original
        image_path = result['image_path']
        original_image = Image.open(image_path).convert('RGB')
        
        # Detección de rostros usando OpenCV
        image_cv = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        
        # Imagen con detección marcada
        image_with_box = image_cv.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
        
        # Extraer rostro recortado
        cropped_face_rgb = None
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_x, center_y = x + w // 2, y + h // 2
            side = max(w, h)
            half_side = side // 2
            
            x1 = max(center_x - half_side, 0)
            y1 = max(center_y - half_side, 0)
            x2 = min(center_x + half_side, image_cv.shape[1])
            y2 = min(center_y + half_side, image_cv.shape[0])
            
            cropped_face = image_cv[y1:y2, x1:x2]
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        
        # Predicciones
        # 1. Imagen original completa
        original_pred = predict_image(model, np.array(original_image), transform, device)
        
        # 2. Rostro detectado
        face_pred = None
        if cropped_face_rgb is not None:
            face_pred = predict_image(model, cropped_face_rgb, transform, device)
        
        # Preprocesar rostro detectado para visualización
        preprocessed_face = None
        if cropped_face_rgb is not None:
            face_pil = Image.fromarray(cropped_face_rgb)
            preprocessed_tensor = transform(face_pil)
            preprocessed_face = preprocessed_tensor.permute(1, 2, 0).numpy()
            
            # Desnormalizar para visualización
            if preprocessed_face.min() < 0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                preprocessed_face = preprocessed_face * std + mean
                preprocessed_face = np.clip(preprocessed_face, 0, 1)
        
        # Visualización en la fila correspondiente
        row = idx
        
        # Columna 1: Imagen original
        axes[row, 0].imshow(original_image)
        axes[row, 0].set_title(f"Original\nReal: {result['true_emotion']}", fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Columna 2: Detección de rostro
        axes[row, 1].imshow(image_with_box_rgb)
        face_status = f"Rostros: {len(faces)}" if len(faces) > 0 else "Sin rostro"
        axes[row, 1].set_title(f"Detección\n{face_status}", fontsize=10, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Columna 3: Rostro recortado
        if cropped_face_rgb is not None:
            axes[row, 2].imshow(cropped_face_rgb)
            axes[row, 2].set_title("Rostro Recortado", fontsize=10, fontweight='bold')
        else:
            axes[row, 2].text(0.5, 0.5, 'No se detectó\nrostro', 
                             ha='center', va='center', fontsize=10)
            axes[row, 2].set_title("Sin Rostro", fontsize=10, fontweight='bold')
        axes[row, 2].axis('off')
        
        # Columna 4: Rostro preprocesado
        if preprocessed_face is not None:
            axes[row, 3].imshow(preprocessed_face)
            pred_emotion = face_pred['predicted_emotion'] if face_pred else 'N/A'
            confidence = face_pred['confidence'] if face_pred else 0
            is_correct = (face_pred and face_pred['predicted_class'] == result['true_class'])
            color = 'green' if is_correct else 'red'
            axes[row, 3].set_title(f"Preprocesado\nPred: {pred_emotion}\nConf: {confidence:.3f}", 
                                  fontsize=10, fontweight='bold', color=color)
        else:
            axes[row, 3].text(0.5, 0.5, 'No disponible', ha='center', va='center', fontsize=10)
            axes[row, 3].set_title("N/A", fontsize=10, fontweight='bold')
        axes[row, 3].axis('off')
        
        # Guardar resultados para comparación
        comparison_result = {
            'image_path': str(image_path),
            'true_emotion': result['true_emotion'],
            'true_class': result['true_class'],
            'original_pred': original_pred,
            'face_detected': len(faces) > 0,
            'face_pred': face_pred,
            'num_faces': len(faces)
        }
        face_detection_results.append(comparison_result)
    
    plt.suptitle('Análisis con Detección de Rostros - Punto 5 TP2\n(Original → Detección → Recortado → Preprocesado)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = results_dir / "face_detection_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Crear reporte comparativo detallado
    report_path = results_dir / "face_detection_comparison.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS CON DETECCIÓN DE ROSTROSn")
        f.write("=" * 70 + "\n\n")
        f.write("Comparación entre imagen completa vs rostro detectado\n")
        f.write("-" * 50 + "\n\n")
        
        correct_original = 0
        correct_face = 0
        total_with_face = 0
        
        for idx, result in enumerate(face_detection_results):
            f.write(f"IMAGEN {idx + 1}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Archivo: {result['image_path']}\n")
            f.write(f"Emoción real: {result['true_emotion']}\n")
            f.write(f"Rostros detectados: {result['num_faces']}\n\n")
            
            # Predicción imagen original
            if result['original_pred']:
                orig_correct = result['original_pred']['predicted_class'] == result['true_class']
                f.write(f"IMAGEN COMPLETA:\n")
                f.write(f"  Predicción: {result['original_pred']['predicted_emotion']}\n")
                f.write(f"  Confianza: {result['original_pred']['confidence']:.4f}\n")
                f.write(f"  Resultado: {'CORRECTO' if orig_correct else 'INCORRECTO'}\n\n")
                if orig_correct:
                    correct_original += 1
            
            # Predicción rostro detectado
            if result['face_detected'] and result['face_pred']:
                face_correct = result['face_pred']['predicted_class'] == result['true_class']
                f.write(f"ROSTRO DETECTADO:\n")
                f.write(f"  Predicción: {result['face_pred']['predicted_emotion']}\n")
                f.write(f"  Confianza: {result['face_pred']['confidence']:.4f}\n")
                f.write(f"  Resultado: {'CORRECTO' if face_correct else 'INCORRECTO'}\n\n")
                if face_correct:
                    correct_face += 1
                total_with_face += 1
            else:
                f.write(f"ROSTRO DETECTADO: No disponible\n\n")
            
            # Scores detallados para rostro detectado
            if result['face_pred']:
                # Obtener scores completos
                original_image = Image.open(result['image_path']).convert('RGB')
                image_cv = cv2.imread(str(result['image_path']))
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    center_x, center_y = x + w // 2, y + h // 2
                    side = max(w, h)
                    half_side = side // 2
                    
                    x1 = max(center_x - half_side, 0)
                    y1 = max(center_y - half_side, 0)
                    x2 = min(center_x + half_side, image_cv.shape[1])
                    y2 = min(center_y + half_side, image_cv.shape[0])
                    
                    cropped_face = image_cv[y1:y2, x1:x2]
                    cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    
                    # Obtener scores detallados
                    with torch.no_grad():
                        face_pil = Image.fromarray(cropped_face_rgb)
                        input_tensor = transform(face_pil).unsqueeze(0).to(device)
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        scores = probabilities.cpu().numpy()[0]
                    
                    f.write("Scores del rostro detectado:\n")
                    for i, emotion in enumerate(EMOTION_LABELS):
                        f.write(f"  {emotion}: {scores[i]:.4f}\n")
            
            f.write("\n" + "="*70 + "\n\n")
        
        # Resumen estadístico
        f.write("RESUMEN COMPARATIVO\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total imágenes analizadas: {len(face_detection_results)}\n")
        f.write(f"Imágenes con rostro detectado: {total_with_face}\n")
        f.write(f"Accuracy imagen completa: {correct_original/len(face_detection_results):.3f}\n")
        if total_with_face > 0:
            f.write(f"Accuracy rostro detectado: {correct_face/total_with_face:.3f}\n")
        else:
            f.write(f"Accuracy rostro detectado: N/A (sin rostros detectados)\n")
        
    
    print(f"Análisis detección rostros guardado: {plot_path}")
    print(f"Reporte comparativo guardado: {report_path}")
    
    return face_detection_results


def plot_affectnet_detailed_results(results, model, transform, device, results_dir):
    """Crea visualización detallada de 6 imágenes de AffectNet con scores y preprocessing"""
    
    # Seleccionar 6 imágenes: una por cada emoción principal, priorizando diversidad
    selected_images = []
    emotions_seen = set()
    
    # Primero añadir una imagen correcta de cada emoción diferente
    for result in results:
        true_emotion = result['true_emotion']
        if (true_emotion not in emotions_seen and 
            result['original_pred'] and 
            result['original_pred']['predicted_class'] == result['true_class']):
            selected_images.append(result)
            emotions_seen.add(true_emotion)
            if len(selected_images) >= 6:
                break
    
    # Si no tenemos 6, completar con imágenes incorrectas
    if len(selected_images) < 6:
        for result in results:
            if (len(selected_images) >= 6):
                break
            if (result not in selected_images and 
                result['original_pred'] and 
                result['original_pred']['predicted_class'] != result['true_class']):
                selected_images.append(result)
    
    # Si aún no tenemos 6, completar con cualquier imagen disponible
    for result in results:
        if len(selected_images) >= 6:
            break
        if result not in selected_images and result['original_pred']:
            selected_images.append(result)
    
    # Crear figura para 6 imágenes (3 filas x 4 columnas: original, preprocesada, scores)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for idx, result in enumerate(selected_images[:6]):
        # Cargar imagen original
        original_image = Image.open(result['image_path']).convert('RGB')
        original_array = np.array(original_image)
        
        # Aplicar transformaciones (preprocessing)
        preprocessed_tensor = transform(original_image)
        preprocessed_image = preprocessed_tensor.permute(1, 2, 0).numpy()
        
        # Normalizar para visualización (las transformaciones incluyen normalización)
        if preprocessed_image.min() < 0:  # Si está normalizado con ImageNet stats
            # Desnormalizar para visualización
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            preprocessed_image = preprocessed_image * std + mean
            preprocessed_image = np.clip(preprocessed_image, 0, 1)
        
        # Obtener scores detallados del modelo
        with torch.no_grad():
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            scores = probabilities.cpu().numpy()[0]
        
        pred_class = np.argmax(scores)
        true_class = result['true_class']
        is_correct = pred_class == true_class
        
        # Calcular posición en la grilla
        row = idx // 2
        col_offset = (idx % 2) * 2
        
        # Mostrar imagen original
        ax_original = axes[row, col_offset]
        ax_original.imshow(original_image)
        ax_original.set_title(f"Original\nReal: {result['true_emotion']}", 
                             fontsize=11, fontweight='bold')
        ax_original.axis('off')
        
        # Mostrar imagen preprocesada
        ax_preprocessed = axes[row, col_offset + 1]
        ax_preprocessed.imshow(preprocessed_image)
        color = 'green' if is_correct else 'red'
        ax_preprocessed.set_title(f"Preprocesada\nPredicho: {EMOTION_LABELS[pred_class]}\nConf: {scores[pred_class]:.3f}", 
                                 fontsize=11, fontweight='bold', color=color)
        ax_preprocessed.axis('off')
        
        # Agregar texto con scores debajo de las imágenes
        scores_text = "\n".join([f"{EMOTION_LABELS[i]}: {scores[i]:.3f}" for i in range(len(EMOTION_LABELS))])
        
        # Posición para el texto de scores
        fig_text_x = 0.02 + col_offset * 0.25
        fig_text_y = 0.85 - row * 0.32
        
        result_text = 'CORRECTO' if is_correct else 'INCORRECTO'
        fig.text(fig_text_x, fig_text_y, f"Scores:\n{scores_text}\n\n{result_text}", 
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
    
    plt.suptitle('Análisis Detallado AffectNet - 6 Imágenes\n(Original, Preprocesada y Scores por Clase)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = results_dir / "affectnet_detailed_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Crear también un reporte detallado en texto
    report_path = results_dir / "affectnet_detailed_scores.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS DETALLADO AFFECTNET - 6 IMÁGENES\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, result in enumerate(selected_images[:6]):
            # Obtener scores del modelo
            original_image = Image.open(result['image_path']).convert('RGB')
            with torch.no_grad():
                input_tensor = transform(original_image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                scores = probabilities.cpu().numpy()[0]
            
            pred_class = np.argmax(scores)
            true_class = result['true_class']
            is_correct = pred_class == true_class
            
            f.write(f"IMAGEN {idx + 1}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Archivo: {result['image_path']}\n")
            f.write(f"Emoción real: {result['true_emotion']}\n")
            f.write(f"Emoción predicha: {EMOTION_LABELS[pred_class]}\n")
            f.write(f"Resultado: {'CORRECTO' if is_correct else 'INCORRECTO'}\n")
            f.write(f"Confianza: {scores[pred_class]:.4f}\n\n")
            
            f.write("Scores por clase:\n")
            for i, emotion in enumerate(EMOTION_LABELS):
                f.write(f"  {emotion}: {scores[i]:.4f}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Análisis detallado AffectNet guardado: {plot_path}")
    print(f"Reporte de scores guardado: {report_path}")
    
    return plot_path, report_path


def plot_affectnet_samples(results, results_dir):
    """Crea visualización de ejemplos de AffectNet"""
    samples = {}
    
    # Seleccionar un ejemplo por emoción
    for result in results:
        true_class = result['true_class']
        true_emotion = result['true_emotion']
        
        if true_emotion in samples:
            continue
            
        if (result['original_pred'] and 
            result['original_pred']['predicted_class'] == true_class):
            samples[true_emotion] = result
            
            if len(samples) == len(EMOTION_LABELS):
                break
    
    # Agregar incorrectas si faltan
    if len(samples) < len(EMOTION_LABELS):
        for result in results:
            true_emotion = result['true_emotion']
            if true_emotion not in samples and result['original_pred']:
                samples[true_emotion] = result
                if len(samples) == len(EMOTION_LABELS):
                    break
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (emotion, result) in enumerate(samples.items()):
        if i >= 7:  # Máximo 7 emociones
            break
            
        image = Image.open(result['image_path']).convert('RGB')
        image_resized = image.resize((224, 224))
        
        pred = result['original_pred']
        true_class = result['true_class']
        pred_class = pred['predicted_class']
        confidence = pred['confidence']
        
        is_correct = pred_class == true_class
        color = 'green' if is_correct else 'red'
        status = 'CORRECTO' if is_correct else 'INCORRECTO'
        
        axes[i].imshow(image_resized)
        axes[i].set_title(f"Real: {emotion}\n"
                         f"Predicho: {pred['predicted_emotion']}\n"
                         f"Confianza: {confidence:.3f}\n"
                         f"{status}", 
                         color=color, fontsize=9, fontweight='bold')
        axes[i].axis('off')
    
    # Ocultar ejes vacíos
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Ejemplos de Clasificación - AffectNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = results_dir / "affectnet_samples.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ejemplos AffectNet guardados: {plot_path}")
    return plot_path



def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generar y guardar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Matriz de confusión guardada: {save_path}")


def plot_class_accuracy(y_true, y_pred, class_names, save_path):
    """Generar gráfico de accuracy por clase"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy, color='skyblue')
    plt.title('Accuracy por Clase')
    plt.xlabel('Emociones')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Accuracy por clase guardado: {save_path}")


def generate_classification_report(y_true, y_pred, class_names, save_path):
    """Generar y guardar reporte de clasificación"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(save_path, 'w') as f:
        f.write("REPORTE DE CLASIFICACIÓN\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\nACCURACY GENERAL\n")
        f.write("=" * 20 + "\n")
        f.write(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    
    print(f"Reporte de clasificación guardado: {save_path}")
    print("\nReporte de Clasificación:")
    print(report)


def evaluate_affectnet(model, config_manager, results_dir):
    """Evalúa el modelo en AffectNet"""
    data_config = config_manager.get_config('data')
    affectnet_path = Path(data_config['affectnet_test_path'])
    
    if not affectnet_path.exists():
        print(f"AffectNet no encontrado en {affectnet_path}")
        print("Para habilitar evaluación AffectNet, coloca el dataset en la estructura:")
        print("  data/test_affectnet/")
        print("    ├── 0_Anger/")
        print("    ├── 1_Disgust/")
        print("    ├── 2_Fear/")
        print("    ├── 3_Happy/")
        print("    ├── 4_Sad/")
        print("    ├── 5_Surprise/")
        print("    └── 6_Neutral/")
        return None
    
    if not verify_test_dataset():
        print("Error: Verificación del dataset falló")
        return None
    
    print("Evaluando en AffectNet...")
    
    # Cargar datos y configurar detección de rostros
    images_data = load_test_affectnet_dataset()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    transform = get_val_transforms()
    device = next(model.parameters()).device
    
    # Evaluar imágenes
    results = []
    for i, image_info in enumerate(images_data):
        image_path = image_info['path']
        
        # Predicción imagen completa
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        original_pred = predict_image(model, original_array, transform, device)
        
        # Predicción con rostro detectado
        face_array = detect_face(image_path, face_cascade)
        face_pred = predict_image(model, face_array, transform, device) if face_array is not None else None
        
        result = {
            'image_path': str(image_path),
            'true_class': image_info['true_class'],
            'true_emotion': image_info['true_emotion'],
            'original_pred': original_pred,
            'face_pred': face_pred,
            'face_detected': face_array is not None
        }
        
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Progreso: {i + 1}/{len(images_data)}")
    
    # Calcular métricas
    true_labels = [r['true_class'] for r in results]
    original_preds = [r['original_pred']['predicted_class'] for r in results if r['original_pred']]
    
    original_accuracy = accuracy_score(true_labels, original_preds)
    
    # Métricas por clase
    class_report = classification_report(true_labels, original_preds, 
                                       target_names=EMOTION_LABELS, 
                                       output_dict=True)
    
    # Calcular accuracy por clase manualmente
    cm = confusion_matrix(true_labels, original_preds)
    class_accuracies = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        if cm[i].sum() > 0:
            class_accuracies[emotion] = cm[i, i] / cm[i].sum()
        else:
            class_accuracies[emotion] = 0.0
    
    # Métricas para rostros detectados
    face_results = [r for r in results if r['face_detected'] and r['face_pred']]
    if face_results:
        face_true = [r['true_class'] for r in face_results]
        face_preds = [r['face_pred']['predicted_class'] for r in face_results]
        face_accuracy = accuracy_score(face_true, face_preds)
    else:
        face_accuracy = 0.0
    
    detection_rate = sum(1 for r in results if r['face_detected']) / len(results)
    
    # Crear plots específicos de AffectNet
    cm = confusion_matrix(true_labels, original_preds)
    
    # Matriz de confusión AffectNet
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Matriz de Confusión - AffectNet')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    
    cm_path = results_dir / "affectnet_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión AffectNet: {cm_path}")
    
    # Ejemplos visuales
    samples_path = plot_affectnet_samples(results, results_dir)
    
    # Análisis detallado de 6 imágenes
    detailed_path, scores_report_path = plot_affectnet_detailed_results(
        results, model, transform, device, results_dir)
    
    # Análisis con detección de rostros - Punto 5 TP2
    face_detection_results = plot_affectnet_face_detection_analysis(
        results, model, transform, device, results_dir)
    
    # Reporte AffectNet
    report_path = results_dir / "affectnet_report.txt"
    with open(report_path, 'w') as f:
        f.write("EVALUACIÓN AFFECTNET\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total imágenes: {len(results)}\n")
        f.write(f"Accuracy imagen completa: {original_accuracy:.4f}\n")
        f.write(f"Accuracy rostros detectados: {face_accuracy:.4f}\n")
        f.write(f"Tasa detección rostros: {detection_rate:.4f}\n")
        f.write(f"Rostros detectados: {sum(1 for r in results if r['face_detected'])}\n\n")
        
        f.write("MÉTRICAS POR CLASE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Emoción':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 60 + "\n")
        
        for emotion in EMOTION_LABELS:
            if emotion in class_report:
                acc = class_accuracies[emotion]
                prec = class_report[emotion]['precision']
                rec = class_report[emotion]['recall']
                f1 = class_report[emotion]['f1-score']
                f.write(f"{emotion:<12} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}\n")
    
    print(f"Reporte AffectNet guardado: {report_path}")
    
    # Mostrar métricas por clase en consola
    print("\nMétricas AffectNet por clase:")
    for emotion in EMOTION_LABELS:
        if emotion in class_report:
            acc = class_accuracies[emotion]
            print(f"  {emotion}: Accuracy={acc:.3f}, Precision={class_report[emotion]['precision']:.3f}, "
                  f"Recall={class_report[emotion]['recall']:.3f}, F1={class_report[emotion]['f1-score']:.3f}")
    
    return {
        'original_accuracy': original_accuracy,
        'face_accuracy': face_accuracy,
        'detection_rate': detection_rate,
        'total_images': len(results),
        'faces_detected': sum(1 for r in results if r['face_detected']),
        'class_accuracies': class_accuracies,
        'classification_report': class_report
    }


def evaluate_model_with_config(config_manager: ConfigManager, model_path: str = None):
    """Evalúa un modelo usando la configuración centralizada"""
    
    # Obtener configuraciones
    data_config = config_manager.get_config('data')
    evaluation_config = config_manager.get_config('evaluation')
    version_info = config_manager.get_config('version_control')
    
    print(f"=== EVALUACIÓN DE MODELO ===")
    print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Directorio de resultados versionado
    results_dir = config_manager.get_results_dir() / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar modelo a evaluar
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"✗ Error: Modelo no encontrado en {model_path}")
            return None
    else:
        # Buscar el modelo más reciente
        models_dir = Path(config_manager.get_config('output')['models_dir'])
        model_files = list(models_dir.glob("*.pth"))
        
        if not model_files:
            print(" Error: No se encontraron modelos entrenados")
            return None
        
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Evaluando modelo: {model_path}")
    
    # Cargar dataset de validación
    val_data_path = Path(data_config['val_path'])
    if not val_data_path.exists():
        print(f" Error: {val_data_path} no existe")
        print("Ejecuta primero: python scripts/create_balanced_dataset.py")
        return None
    
    val_dataset = EmotionDataset(str(val_data_path), transform=get_val_transforms())
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    # Cargar modelo usando la clase Evaluator
    try:
        model, checkpoint = Evaluator.load_model(str(model_path), device)
        print(" Modelo cargado exitosamente")
        
        # Obtener información del checkpoint si está disponible
        model_info = {}
        if checkpoint:
            model_info = {
                'epoch': int(checkpoint.get('epoch', 0)) if checkpoint.get('epoch', 'N/A') != 'N/A' else 'N/A',
                'val_acc': float(checkpoint.get('val_acc', 0.0)) if checkpoint.get('val_acc', 'N/A') != 'N/A' else 'N/A',
                'config_hash': str(checkpoint.get('config_hash', 'N/A')),
                'experiment_id': str(checkpoint.get('experiment_id', 'N/A'))
            }
            print(f"  Época: {model_info['epoch']}")
            if isinstance(model_info['val_acc'], float):
                print(f"  Val Accuracy (entrenamiento): {model_info['val_acc']:.4f}")
            print(f"  Config Hash: {model_info['config_hash']}")
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        return None
    
    # Crear evaluador
    evaluator = Evaluator(model, val_loader, device)
    
    # Evaluar modelo
    print("Evaluando modelo...")
    results = evaluator.get_detailed_results()
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    class_names = results['class_names']
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    
    # Matriz de confusión
    confusion_matrix_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(true_labels, predictions, class_names, confusion_matrix_path)
    
    # Accuracy por clase
    class_accuracy_path = results_dir / "class_accuracy.png"
    plot_class_accuracy(true_labels, predictions, class_names, class_accuracy_path)
    
    # Reporte de clasificación
    report_path = results_dir / "classification_report.txt"
    generate_classification_report(true_labels, predictions, class_names, report_path)
    
    # Calcular métricas adicionales
    overall_accuracy = results['overall_accuracy']
    class_accuracies = results['class_accuracies']
    
    # Calcular F1 Score
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    print(f"\n=== MÉTRICAS DE EVALUACIÓN ===")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    
    # Guardar resultados completos
    evaluation_results = {
        "timestamp": config_manager.timestamp,
        "model_info": {
            "model_path": str(model_path),
            "checkpoint_info": model_info
        },
        "dataset_info": {
            "val_path": str(val_data_path),
            "total_samples": int(len(val_dataset)),
            "num_classes": int(NUM_CLASSES)
        },
        "metrics": {
            "overall_accuracy": float(overall_accuracy),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "f1_per_class": {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))},
            "class_accuracies": {k: float(v) for k, v in class_accuracies.items()}
        },
        "evaluation_config": evaluation_config,
        "predictions_summary": {
            "total_predictions": int(len(predictions)),
            "correct_predictions": int(sum(p == t for p, t in zip(predictions, true_labels)))
        }
    }
    
    # Guardar resultados en JSON
    results_json_path = results_dir / "evaluation_results.json"
    # Convertir tipos numpy antes de serializar
    evaluation_results_clean = convert_numpy_types(evaluation_results)
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results_clean, f, indent=2, ensure_ascii=False)
    
    # Evaluar en AffectNet si está disponible
    affectnet_results = evaluate_affectnet(model, config_manager, results_dir)
    if affectnet_results:
        evaluation_results['affectnet_metrics'] = affectnet_results
        
        # Actualizar JSON con métricas de AffectNet
        evaluation_results_clean = convert_numpy_types(evaluation_results)
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results_clean, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print(f"\n=== RESULTADOS DE EVALUACIÓN ===")
    print(f" Accuracy general: {overall_accuracy:.4f}")
    print(f" Muestras evaluadas: {len(predictions)}")
    print(f" Predicciones correctas: {evaluation_results['predictions_summary']['correct_predictions']}")
    
    print("\nAccuracy por clase:")
    for emotion, acc in class_accuracies.items():
        print(f"  {emotion}: {acc:.4f}")
    
    # Mostrar resultados de AffectNet si están disponibles
    if affectnet_results:
        print(f"\n=== RESULTADOS AFFECTNET ===")
        print(f" Accuracy imagen completa: {affectnet_results['original_accuracy']:.4f}")
        print(f" Accuracy rostros detectados: {affectnet_results['face_accuracy']:.4f}")
        print(f" Tasa detección rostros: {affectnet_results['detection_rate']:.4f}")
        print(f" Imágenes evaluadas: {affectnet_results['total_images']}")
        print(f" Rostros detectados: {affectnet_results['faces_detected']}")
    
    print(f"\n Resultados guardados en: {results_dir}")
    print(f" Archivo de resultados: {results_json_path}")
    
    return evaluation_results


def main():
    """Función principal"""
    print("=== EVALUADOR DE MODELO ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager()
        print(" ConfigManager inicializado")
        
        # Evaluar modelo
        results = evaluate_model_with_config(config_manager)
        
        if results:
            print("\n Evaluación completada exitosamente")
        else:
            print("\n Evaluación falló")
            
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
