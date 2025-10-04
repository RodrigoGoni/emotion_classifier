"""
Evaluación completa del modelo con AffectNet integrado con sistema de versionado
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
import json
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import CNNModel
from src.utils.constants import EMOTION_CLASSES, EMOTION_LABELS, NUM_CLASSES
from src.data.transforms import get_val_transforms
from src.utils.test_dataset import load_test_affectnet_dataset, verify_test_dataset
from src.utils.config_manager import ConfigManager


class AffectNetEvaluator:
    def __init__(self, config_manager: ConfigManager, model_path=None):
        self.config_manager = config_manager
        self.data_config = config_manager.get_config('data')
        self.inference_config = config_manager.get_config('inference')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = get_val_transforms()
        
        # Configuración de detección de rostros desde config
        cascade_path = self.inference_config.get('face_detection', {}).get('cascade_path', 
                                                'haarcascade_frontalface_default.xml')
        if not Path(cascade_path).exists():
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = self.inference_config.get('face_detection', {}).get('scale_factor', 1.1)
        self.min_neighbors = self.inference_config.get('face_detection', {}).get('min_neighbors', 6)
    
    def load_model(self, model_path=None):
        if model_path is None:
            models_dir = Path(self.config_manager.get_config('output')['models_dir'])
            model_files = list(models_dir.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No se encontraron modelos entrenados")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Usar configuración desde config_manager
        model_config = self.config_manager.get_config('model')
        model = CNNModel(
            num_classes=NUM_CLASSES,
            input_size=(self.data_config['image_size'], self.data_config['image_size']),
            num_channels=self.data_config['channels'],
            dropout_prob=model_config['dropout_prob']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Modelo cargado: {model_path}")
        if 'config_hash' in checkpoint:
            print(f"Config Hash: {checkpoint['config_hash']}")
        if 'experiment_id' in checkpoint:
            print(f"Experiment ID: {checkpoint['experiment_id']}")
        
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
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor, 
            minNeighbors=self.min_neighbors
        )
        
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


def run_evaluation_with_config():
    """Función principal de evaluación con sistema de versionado"""
    print("=== EVALUACIÓN AFFECTNET CON VERSIONADO ===")
    
    try:
        # Inicializar ConfigManager
        config_manager = ConfigManager()
        print("ConfigManager inicializado")
        
        # Obtener configuraciones
        version_info = config_manager.get_config('version_control')
        evaluation_config = config_manager.get_config('evaluation')
        data_config = config_manager.get_config('data')
        
        print(f"Proyecto: {version_info['project_name']} v{version_info['version']}")
        
        # Configurar MLflow con versionado
        experiment_id = config_manager.setup_mlflow_experiment()
        run_name = f"affectnet_eval_{config_manager.timestamp}"
        run_id = config_manager.start_mlflow_run(run_name)
        
        print(f"Experimento MLflow: {experiment_id}")
        print(f"Run ID: {run_id}")
        
        # Verificar dataset AffectNet
        affectnet_path = Path(data_config['affectnet_test_path'])
        if not affectnet_path.exists():
            print(f"Error: Dataset AffectNet no encontrado en {affectnet_path}")
            print("Por favor, asegúrate de que el dataset esté disponible")
            return None
        
        print(f"Dataset AffectNet encontrado: {affectnet_path}")
        
        # Verificar dataset usando la función existente
        if not verify_test_dataset():
            print("Error: Verificación del dataset falló")
            return None
        
        # Cargar datos
        images_data = load_test_affectnet_dataset()
        print(f"Dataset cargado: {len(images_data)} imágenes")
        
        # Log configuración del dataset
        mlflow.log_param("affectnet_path", str(affectnet_path))
        mlflow.log_param("dataset_size", len(images_data))
        
        # Crear evaluador con configuración
        evaluator = AffectNetEvaluator(config_manager)
        
        # Log información del modelo
        model_info = evaluator.model[1]  # checkpoint
        model_path = evaluator.model[2]
        
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("model_config_hash", model_info.get('config_hash', 'N/A'))
        mlflow.log_param("model_experiment_id", model_info.get('experiment_id', 'N/A'))
        mlflow.log_param("train_accuracy", model_info.get('val_acc', 0))
        
        # Log configuración de detección de rostros
        inference_config = config_manager.get_config('inference')
        face_config = inference_config.get('face_detection', {})
        mlflow.log_param("face_scale_factor", face_config.get('scale_factor', 1.1))
        mlflow.log_param("face_min_neighbors", face_config.get('min_neighbors', 6))
        
        # Log nombres de clases
        for i, emotion in enumerate(EMOTION_LABELS):
            mlflow.log_param(f"class_{i}", emotion)
        
        # Evaluar modelo
        print("Evaluando modelo en AffectNet...")
        results = evaluator.evaluate_dataset(images_data)
        
        # Calcular métricas
        print("Calculando métricas...")
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
        
        # Crear directorio de resultados versionado
        results_dir = config_manager.get_results_dir() / "affectnet_evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generando visualizaciones...")
        
        # Crear plots
        cm_path = evaluator.plot_confusion_matrix(metrics['confusion_matrix'], results_dir)
        samples_path = evaluator.plot_sample_results(results, results_dir)
        
        # Log artifacts
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(samples_path))
        
        # Crear reporte detallado con versionado
        report_data = {
            "timestamp": config_manager.timestamp,
            "experiment_info": {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "config_hash": getattr(config_manager, '_config_hash', 'N/A')
            },
            "model_info": {
                "model_path": str(model_path),
                "model_config_hash": model_info.get('config_hash', 'N/A'),
                "model_experiment_id": model_info.get('experiment_id', 'N/A'),
                "train_accuracy": model_info.get('val_acc', 'N/A')
            },
            "dataset_info": {
                "affectnet_path": str(affectnet_path),
                "total_images": metrics['total_images'],
                "faces_detected": metrics['faces_detected']
            },
            "metrics": {
                "accuracy_original": metrics['original_accuracy'],
                "accuracy_faces": metrics['face_accuracy'],
                "detection_rate": metrics['detection_rate'],
                "classification_report": metrics['classification_report']
            },
            "configuration": {
                "version_control": version_info,
                "evaluation_config": evaluation_config,
                "inference_config": inference_config
            }
        }
        
        # Guardar reporte JSON
        report_json_path = results_dir / "affectnet_evaluation_report.json"
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar reporte de texto
        report_txt_path = results_dir / "affectnet_evaluation_report.txt"
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write("EVALUACIÓN AFFECTNET CON VERSIONADO\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("INFORMACIÓN DEL EXPERIMENTO:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Timestamp: {config_manager.timestamp}\n")
            f.write(f"Proyecto: {version_info['project_name']} v{version_info['version']}\n")
            f.write(f"Experiment ID: {experiment_id}\n")
            f.write(f"Run ID: {run_id}\n\n")
            
            f.write("INFORMACIÓN DEL MODELO:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Modelo: {model_path}\n")
            f.write(f"Config Hash: {model_info.get('config_hash', 'N/A')}\n")
            f.write(f"Accuracy entrenamiento: {model_info.get('val_acc', 'N/A')}\n\n")
            
            f.write("MÉTRICAS GENERALES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total imágenes: {metrics['total_images']}\n")
            f.write(f"Accuracy imagen completa: {metrics['original_accuracy']:.4f}\n")
            f.write(f"Accuracy rostros detectados: {metrics['face_accuracy']:.4f}\n")
            f.write(f"Tasa detección rostros: {metrics['detection_rate']:.4f}\n")
            f.write(f"Rostros detectados: {metrics['faces_detected']}\n\n")
            
            f.write("MÉTRICAS POR EMOCIÓN:\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Emoción':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 45 + "\n")
            
            for emotion in EMOTION_LABELS:
                if emotion in metrics['classification_report']:
                    report = metrics['classification_report'][emotion]
                    f.write(f"{emotion:<12} {report['precision']:<10.3f} "
                           f"{report['recall']:<10.3f} {report['f1-score']:<10.3f}\n")
            
            f.write(f"\nARCHIVOS GENERADOS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Matriz de confusión: {cm_path.name}\n")
            f.write(f"Ejemplos visuales: {samples_path.name}\n")
            f.write(f"Reporte JSON: {report_json_path.name}\n")
        
        # Log reportes
        mlflow.log_artifact(str(report_json_path))
        mlflow.log_artifact(str(report_txt_path))
        
        # Guardar resumen del experimento
        config_manager.save_experiment_summary(
            {
                'affectnet_accuracy_original': metrics['original_accuracy'],
                'affectnet_accuracy_faces': metrics['face_accuracy'],
                'affectnet_detection_rate': metrics['detection_rate'],
                'affectnet_total_images': metrics['total_images']
            },
            str(model_path),
            {
                'evaluation_type': 'affectnet',
                'dataset_path': str(affectnet_path),
                'faces_detected': metrics['faces_detected']
            }
        )
        
        # Finalizar run de MLflow
        mlflow.end_run()
        
        # Mostrar resumen
        print(f"\n=== RESULTADOS DE EVALUACIÓN AFFECTNET ===")
        print(f"Accuracy imagen completa: {metrics['original_accuracy']:.4f}")
        print(f"Accuracy rostros detectados: {metrics['face_accuracy']:.4f}")
        print(f"Tasa detección rostros: {metrics['detection_rate']:.4f}")
        print(f"Total imágenes evaluadas: {metrics['total_images']}")
        print(f"Rostros detectados: {metrics['faces_detected']}")
        
        print(f"\nResultados guardados en: {results_dir}")
        print(f"Reporte detallado: {report_txt_path}")
        print(f"Datos JSON: {report_json_path}")
        print(f"Experimento MLflow: {run_id}")
        
        return metrics
        
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        if mlflow.active_run():
            mlflow.end_run(status='FAILED')
        import traceback
        traceback.print_exc()
        return None


# Mantener función original para compatibilidad
def run_evaluation():
    """Función de evaluación original (deprecated)"""
    print("⚠️  Usando función de evaluación legacy. Se recomienda usar run_evaluation_with_config()")
    return run_evaluation_with_config()


if __name__ == "__main__":
    run_evaluation_with_config()