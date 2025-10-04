"""
Evaluación completa del modelo con AffectNet integrado con MLOps
"""
import os
import sys
import torch
import cv2
import numpy as np
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import CNNModel
from src.utils.constants import EMOTION_CLASSES, EMOTION_LABELS, NUM_CLASSES, IMAGE_SIZE, CHANNELS
from src.data.transforms import get_val_transforms
from src.utils.test_dataset import load_test_affectnet_dataset, verify_test_dataset


class AffectNetEvaluator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = get_val_transforms()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def load_model(self, model_path=None):
        if model_path is None:
            models_dir = Path("models/trained")
            model_files = list(models_dir.glob("best_model_epoch_*.pth"))
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CNNModel(
            num_classes=NUM_CLASSES,
            input_size=(IMAGE_SIZE, IMAGE_SIZE),
            num_channels=CHANNELS,
            dropout_prob=0.5
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Modelo cargado: {model_path}")
        return model, checkpoint, model_path
    
    def predict_image(self, image_array):
        if image_array is None:
            return None
        
        pil_image = Image.fromarray(image_array)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model[0](image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            scores = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(scores)
            confidence = scores[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_emotion': EMOTION_CLASSES[predicted_class],
            'confidence': confidence,
            'scores': scores
        }
    
    def detect_face(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
        
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
    
    def evaluate_dataset(self, images_data):
        results = []
        
        for i, image_info in enumerate(images_data):
            image_path = image_info['path']
            
            # Predicción imagen completa
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image)
            original_pred = self.predict_image(original_array)
            
            # Predicción con rostro detectado
            face_array = self.detect_face(image_path)
            face_pred = self.predict_image(face_array) if face_array is not None else None
            
            result = {
                'image_path': str(image_path),
                'true_class': image_info['true_class'],
                'true_emotion': image_info['true_emotion'],
                'affectnet_name': image_info['affectnet_name'],
                'original_pred': original_pred,
                'face_pred': face_pred,
                'face_detected': face_array is not None
            }
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Progreso: {i + 1}/{len(images_data)}")
        
        return results
    
    def calculate_metrics(self, results):
        # Métricas para imagen completa
        true_labels = [r['true_class'] for r in results]
        original_preds = [r['original_pred']['predicted_class'] for r in results if r['original_pred']]
        
        original_accuracy = accuracy_score(true_labels, original_preds)
        
        # Métricas para rostros detectados
        face_results = [r for r in results if r['face_detected'] and r['face_pred']]
        if face_results:
            face_true = [r['true_class'] for r in face_results]
            face_preds = [r['face_pred']['predicted_class'] for r in face_results]
            face_accuracy = accuracy_score(face_true, face_preds)
        else:
            face_accuracy = 0.0
        
        detection_rate = sum(1 for r in results if r['face_detected']) / len(results)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, original_preds)
        
        # Classification report con nombres de emociones
        class_report = classification_report(true_labels, original_preds, 
                                           target_names=EMOTION_LABELS, 
                                           output_dict=True)
        
        return {
            'original_accuracy': original_accuracy,
            'face_accuracy': face_accuracy,
            'detection_rate': detection_rate,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'total_images': len(results),
            'faces_detected': sum(1 for r in results if r['face_detected'])
        }
    
    def plot_confusion_matrix(self, cm, results_dir):
        """Crear matriz de confusión"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=EMOTION_LABELS, 
                   yticklabels=EMOTION_LABELS)
        plt.title('Matriz de Confusión - AffectNet')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        plot_path = results_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_sample_results(self, results, results_dir):
        """Crear visualización de 6 ejemplos (uno por emoción)"""
        # Seleccionar un ejemplo correcto de cada emoción
        samples = {}
        
        for result in results:
            true_class = result['true_class']
            true_emotion = result['true_emotion']
            
            # Si ya tenemos muestra de esta emoción, continuar
            if true_emotion in samples:
                continue
                
            # Verificar si la predicción es correcta
            if (result['original_pred'] and 
                result['original_pred']['predicted_class'] == true_class):
                samples[true_emotion] = result
                
                # Si tenemos 6 emociones, terminar
                if len(samples) == len(EMOTION_CLASSES):
                    break
        
        # Si no tenemos suficientes correctas, agregar incorrectas
        if len(samples) < len(EMOTION_LABELS):
            for result in results:
                true_emotion = result['true_emotion']
                if true_emotion not in samples and result['original_pred']:
                    samples[true_emotion] = result
                    if len(samples) == len(EMOTION_LABELS):
                        break
        
        # Crear plot con 2 filas x 3 columnas
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (emotion, result) in enumerate(samples.items()):
            if i >= 6:
                break
                
            # Cargar imagen
            image = Image.open(result['image_path']).convert('RGB')
            image_resized = image.resize((224, 224))
            
            # Información de predicción
            pred = result['original_pred']
            true_class = result['true_class']
            pred_class = pred['predicted_class']
            confidence = pred['confidence']
            
            # Determinar si es correcto
            is_correct = pred_class == true_class
            color = 'green' if is_correct else 'red'
            status = 'CORRECTO' if is_correct else 'INCORRECTO'
            
            # Mostrar imagen
            axes[i].imshow(image_resized)
            axes[i].set_title(f"Real: {emotion}\n"
                            f"Predicho: {pred['predicted_emotion']}\n"
                            f"Confianza: {confidence:.3f}\n"
                            f"Estado: {status}", 
                            color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Ejemplos de Clasificación por Emoción', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = results_dir / "sample_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path


def run_evaluation():
    # Configurar MLflow
    mlflow.set_experiment("affectnet_evaluation")
    
    with mlflow.start_run():
        # Verificar dataset
        if not verify_test_dataset():
            print("Error: Dataset de test no encontrado")
            return
        
        # Cargar datos
        images_data = load_test_affectnet_dataset()
        print(f"Dataset cargado: {len(images_data)} imágenes")
        
        # Crear evaluador
        evaluator = AffectNetEvaluator()
        
        # Log model info y nombres de clases
        model_info = evaluator.model[1]  # checkpoint
        mlflow.log_param("model_path", str(evaluator.model[2]))
        mlflow.log_param("train_accuracy", model_info.get('val_acc', 0))
        mlflow.log_param("dataset_size", len(images_data))
        
        # Log nombres de clases
        for i, emotion in enumerate(EMOTION_LABELS):
            mlflow.log_param(f"class_{i}", emotion)
        
        # Evaluar
        print("Evaluando modelo...")
        results = evaluator.evaluate_dataset(images_data)
        
        # Calcular métricas
        metrics = evaluator.calculate_metrics(results)
        
        # Log métricas generales a MLflow
        mlflow.log_metric("accuracy_original", metrics['original_accuracy'])
        mlflow.log_metric("accuracy_faces", metrics['face_accuracy'])
        mlflow.log_metric("face_detection_rate", metrics['detection_rate'])
        mlflow.log_metric("total_images", metrics['total_images'])
        mlflow.log_metric("faces_detected", metrics['faces_detected'])
        
        # Log métricas por clase
        for emotion in EMOTION_LABELS:
            if emotion in metrics['classification_report']:
                precision = metrics['classification_report'][emotion]['precision']
                recall = metrics['classification_report'][emotion]['recall']
                f1_score = metrics['classification_report'][emotion]['f1-score']
                
                mlflow.log_metric(f"precision_{emotion}", precision)
                mlflow.log_metric(f"recall_{emotion}", recall)
                mlflow.log_metric(f"f1_score_{emotion}", f1_score)
        
        # Crear directorio de resultados
        results_dir = Path("results/evaluation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear plots
        print("Generando visualizaciones...")
        
        # Matriz de confusión
        cm_path = evaluator.plot_confusion_matrix(metrics['confusion_matrix'], results_dir)
        mlflow.log_artifact(str(cm_path))
        
        # Ejemplos de resultados
        samples_path = evaluator.plot_sample_results(results, results_dir)
        mlflow.log_artifact(str(samples_path))
        
        # Reporte detallado
        report_path = results_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EVALUACION AFFECTNET\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METRICAS GENERALES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total imágenes: {metrics['total_images']}\n")
            f.write(f"Accuracy imagen completa: {metrics['original_accuracy']:.3f}\n")
            f.write(f"Accuracy rostros detectados: {metrics['face_accuracy']:.3f}\n")
            f.write(f"Tasa detección rostros: {metrics['detection_rate']:.3f}\n")
            f.write(f"Rostros detectados: {metrics['faces_detected']}\n\n")
            
            f.write("METRICAS POR EMOCION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Emoción':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 45 + "\n")
            
            for i, emotion in enumerate(EMOTION_LABELS):
                if emotion in metrics['classification_report']:
                    report = metrics['classification_report'][emotion]
                    f.write(f"{emotion:<12} {report['precision']:<10.3f} "
                           f"{report['recall']:<10.3f} {report['f1-score']:<10.3f}\n")
            
            f.write(f"\nMATRIZ DE CONFUSION:\n")
            f.write("-" * 20 + "\n")
            f.write("Ver archivo: confusion_matrix.png\n\n")
            
            f.write("EJEMPLOS VISUALES:\n")
            f.write("-" * 18 + "\n")
            f.write("Ver archivo: sample_results.png\n")
        
        # Log reporte a MLflow
        mlflow.log_artifact(str(report_path))
        
        print(f"\nResultados:")
        print(f"Accuracy original: {metrics['original_accuracy']:.3f}")
        print(f"Accuracy rostros: {metrics['face_accuracy']:.3f}")
        print(f"Detección rostros: {metrics['detection_rate']:.3f}")
        print(f"Reporte guardado: {report_path}")
        print(f"Visualizaciones: {results_dir}")
        
        return metrics


if __name__ == "__main__":
    run_evaluation()