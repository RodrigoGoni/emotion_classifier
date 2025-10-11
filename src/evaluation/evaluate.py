"""
Módulo de evaluación para modelos entrenados
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.models.cnn_model import CNNModel
from src.utils.constants import *
from src.data.transforms import get_val_transforms
from src.utils.test_dataset import load_test_affectnet_dataset, verify_test_dataset


class Evaluator:
    """Clase para evaluar modelos entrenados"""
    
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
    
    @staticmethod
    def load_model(model_path, device):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            conv_layers = model_config.get('conv_layers')
            fc_layers = model_config.get('fc_layers') 
            dropout_prob = model_config.get('dropout_prob', 0.5)
            
            if conv_layers is None:
                conv_layers = [
                    {'filters': 64, 'kernel': 3, 'pool': 2},
                    {'filters': 128, 'kernel': 3, 'pool': 2},
                    {'filters': 256, 'kernel': 3, 'pool': 2},
                    {'filters': 512, 'kernel': 3, 'pool': 2}
                ]
            
            if fc_layers is None:
                state_dict = checkpoint['model_state_dict']
                fc_layers = []
                classifier_layers = {}
                for name, tensor in state_dict.items():
                    if 'classifier' in name and 'weight' in name:
                        layer_num = int(name.split('.')[1])
                        classifier_layers[layer_num] = tensor.shape
                
                sorted_layers = sorted(classifier_layers.items())
                for i, (layer_num, shape) in enumerate(sorted_layers[:-1]):
                    fc_layers.append(shape[0])
            
            model = CNNModel(
                num_classes=NUM_CLASSES,
                input_size=(IMAGE_SIZE, IMAGE_SIZE),
                num_channels=CHANNELS,
                dropout_prob=dropout_prob,
                conv_layers=conv_layers,
                fc_layers=fc_layers
            )
        else:
            model = CNNModel(
                num_classes=NUM_CLASSES,
                input_size=(IMAGE_SIZE, IMAGE_SIZE),
                num_channels=CHANNELS,
                dropout_prob=0.5
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, checkpoint
    
    @staticmethod
    def convert_numpy_types(obj):
        """Convierte recursivamente tipos numpy a tipos Python nativos para JSON"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: Evaluator.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [Evaluator.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def predict_image(self, image_array, transform):
        """Predice emoción de una imagen"""
        if image_array is None:
            return None
        
        pil_image = Image.fromarray(image_array)
        image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            scores = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(scores)
            confidence = scores[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_emotion': EMOTION_LABELS[predicted_class],
            'confidence': confidence,
            'scores': scores
        }
    
    def detect_face(self, image_path):
        """Detecta y extrae rostro de una imagen"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path):
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
    
    def plot_class_accuracy(self, y_true, y_pred, class_names, save_path):
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
        
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_affectnet(self, config_manager, results_dir):
        """Evalúa el modelo en AffectNet y retorna métricas completas"""
        data_config = config_manager.get_config('data')
        affectnet_path = Path(data_config['affectnet_test_path'])
        
        if not affectnet_path.exists() or not verify_test_dataset():
            return None
        
        images_data = load_test_affectnet_dataset()
        transform = get_val_transforms()
        
        results = []
        for image_info in images_data:
            image_path = image_info['path']
            
            # Predicción imagen completa
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image)
            original_pred = self.predict_image(original_array, transform)
            
            # Predicción con rostro detectado
            face_array = self.detect_face(image_path)
            face_pred = self.predict_image(face_array, transform) if face_array is not None else None
            
            result = {
                'image_path': str(image_path),
                'true_class': image_info['true_class'],
                'true_emotion': image_info['true_emotion'],
                'original_pred': original_pred,
                'face_pred': face_pred,
                'face_detected': face_array is not None
            }
            results.append(result)
        
        # Calcular métricas
        true_labels = [r['true_class'] for r in results]
        original_preds = [r['original_pred']['predicted_class'] for r in results if r['original_pred']]
        
        original_accuracy = accuracy_score(true_labels, original_preds)
        class_report = classification_report(true_labels, original_preds, 
                                           target_names=EMOTION_LABELS, 
                                           output_dict=True)
        
        # Métricas para rostros detectados
        face_results = [r for r in results if r['face_detected'] and r['face_pred']]
        if face_results:
            face_true = [r['true_class'] for r in face_results]
            face_preds = [r['face_pred']['predicted_class'] for r in face_results]
            face_accuracy = accuracy_score(face_true, face_preds)
        else:
            face_accuracy = 0.0
        
        detection_rate = sum(1 for r in results if r['face_detected']) / len(results)
        
        # Crear visualizaciones
        self.create_affectnet_visualization(results, results_dir)
        
        return {
            'original_accuracy': original_accuracy,
            'face_accuracy': face_accuracy,
            'detection_rate': detection_rate,
            'total_images': len(results),
            'faces_detected': sum(1 for r in results if r['face_detected']),
            'class_report': class_report,
            'detailed_results': results[:6]  # Solo primeras 6 para reporte
        }
    
    def create_affectnet_visualization(self, results, results_dir):
        """Crea visualización combinada de AffectNet"""
        # Seleccionar 6 imágenes representativas
        selected_images = []
        emotions_seen = set()
        
        for result in results:
            true_emotion = result['true_emotion']
            if (true_emotion not in emotions_seen and 
                result['original_pred'] and len(selected_images) < 6):
                selected_images.append(result)
                emotions_seen.add(true_emotion)
        
        # Completar hasta 6 si es necesario
        for result in results:
            if len(selected_images) >= 6:
                break
            if result not in selected_images and result['original_pred']:
                selected_images.append(result)
        
        # Crear figura con análisis completo
        fig, axes = plt.subplots(6, 4, figsize=(20, 24))
        transform = get_val_transforms()
        
        for idx, result in enumerate(selected_images[:6]):
            # Cargar y procesar imagen
            image_path = result['image_path']
            original_image = Image.open(image_path).convert('RGB')
            
            # Detección de rostros
            image_cv = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
            
            # Imagen con detección marcada
            image_with_box = image_cv.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
            
            # Rostro recortado
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
            
            # Rostro preprocesado
            preprocessed_face = None
            if cropped_face_rgb is not None:
                face_pil = Image.fromarray(cropped_face_rgb)
                preprocessed_tensor = transform(face_pil)
                preprocessed_face = preprocessed_tensor.permute(1, 2, 0).numpy()
                
                if preprocessed_face.min() < 0:
                    preprocessed_face = (preprocessed_face - preprocessed_face.min()) / (preprocessed_face.max() - preprocessed_face.min())
            
            # Visualizar en fila
            row = idx
            
            # Original
            axes[row, 0].imshow(original_image)
            axes[row, 0].set_title(f"Original\nReal: {result['true_emotion']}", fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')
            
            # Detección
            axes[row, 1].imshow(image_with_box_rgb)
            face_status = f"Rostros: {len(faces)}" if len(faces) > 0 else "Sin rostro"
            axes[row, 1].set_title(f"Detección\n{face_status}", fontsize=10, fontweight='bold')
            axes[row, 1].axis('off')
            
            # Recortado
            if cropped_face_rgb is not None:
                axes[row, 2].imshow(cropped_face_rgb)
                axes[row, 2].set_title("Rostro Recortado", fontsize=10, fontweight='bold')
            else:
                axes[row, 2].text(0.5, 0.5, 'No se detectó\nrostro', ha='center', va='center', fontsize=10)
                axes[row, 2].set_title("Sin Rostro", fontsize=10, fontweight='bold')
            axes[row, 2].axis('off')
            
            # Preprocesado
            if preprocessed_face is not None:
                axes[row, 3].imshow(preprocessed_face)
                face_pred = result['face_pred']
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
        
        plt.suptitle('Análisis AffectNet - Detección de Rostros\n(Original → Detección → Recortado → Preprocesado)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = results_dir / "affectnet_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def evaluate_model(self):
        """Evaluar modelo y obtener predicciones y etiquetas verdaderas"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities)
        }
    
    def get_accuracy(self):
        """Calcular accuracy del modelo"""
        results = self.evaluate_model()
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        correct = (predictions == true_labels).sum()
        total = len(true_labels)
        accuracy = correct / total
        
        return accuracy
    
    def get_class_accuracies(self):
        """Calcular accuracy por cada clase"""
        results = self.evaluate_model()
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        class_accuracies = {}
        for class_idx in range(NUM_CLASSES):
            class_mask = (true_labels == class_idx)
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_true = true_labels[class_mask]
                class_accuracy = (class_predictions == class_true).sum() / len(class_true)
                class_accuracies[EMOTION_CLASSES[class_idx]] = class_accuracy
            else:
                class_accuracies[EMOTION_CLASSES[class_idx]] = 0.0
        
        return class_accuracies
    
    def get_detailed_results(self):
        """Obtener resultados detallados para análisis posterior"""
        results = self.evaluate_model()
        
        return {
            'predictions': results['predictions'],
            'true_labels': results['true_labels'],
            'probabilities': results['probabilities'],
            'overall_accuracy': self.get_accuracy(),
            'class_accuracies': self.get_class_accuracies(),
            'num_samples': len(results['true_labels']),
            'num_classes': NUM_CLASSES,
            'class_names': [EMOTION_CLASSES[i] for i in range(NUM_CLASSES)]
        }
    
    def generate_complete_report(self, config_manager, model_path, results_dir):
        """Genera un reporte completo consolidado"""
        # Obtener resultados básicos
        results = self.get_detailed_results()
        predictions = results['predictions']
        true_labels = results['true_labels']
        class_names = results['class_names']
        
        # Métricas básicas
        overall_accuracy = results['overall_accuracy']
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # Generar visualizaciones
        self.plot_confusion_matrix(true_labels, predictions, class_names, 
                                 results_dir / "confusion_matrix.png")
        self.plot_class_accuracy(true_labels, predictions, class_names, 
                               results_dir / "class_accuracy.png")
        
        # Evaluar AffectNet
        affectnet_results = self.evaluate_affectnet(config_manager, results_dir)
        
        # Crear reporte consolidado
        report_path = results_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPLETO DE EVALUACIÓN\n")
            f.write("=" * 80 + "\n\n")
            
            # Información del modelo
            version_info = config_manager.get_config('version_control')
            f.write(f"Proyecto: {version_info['project_name']} v{version_info['version']}\n")
            f.write(f"Timestamp: {config_manager.timestamp}\n")
            f.write(f"Modelo: {model_path}\n\n")
            
            # Métricas generales
            f.write("MÉTRICAS GENERALES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy general: {overall_accuracy:.4f}\n")
            f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
            f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
            f.write(f"Total muestras: {len(predictions)}\n")
            f.write(f"Predicciones correctas: {sum(p == t for p, t in zip(predictions, true_labels))}\n\n")
            
            # Métricas por clase
            f.write("MÉTRICAS POR CLASE\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Emoción':<12} {'Accuracy':<10} {'F1-Score':<10}\n")
            f.write("-" * 35 + "\n")
            
            class_accuracies = results['class_accuracies']
            for i, emotion in enumerate(class_names):
                emotion_key = EMOTION_CLASSES[i]
                acc = class_accuracies.get(emotion_key, 0.0)
                f1 = f1_per_class[i]
                f.write(f"{emotion:<12} {acc:<10.4f} {f1:<10.4f}\n")
            
            # Reporte de clasificación sklearn
            f.write("\nREPORTE DETALLADO DE CLASIFICACIÓN\n")
            f.write("-" * 40 + "\n")
            class_report = classification_report(true_labels, predictions, target_names=class_names)
            f.write(class_report)
            
            # Resultados AffectNet
            if affectnet_results:
                f.write("\n\nEVALUACIÓN AFFECTNET\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total imágenes: {affectnet_results['total_images']}\n")
                f.write(f"Accuracy imagen completa: {affectnet_results['original_accuracy']:.4f}\n")
                f.write(f"Accuracy rostros detectados: {affectnet_results['face_accuracy']:.4f}\n")
                f.write(f"Tasa detección rostros: {affectnet_results['detection_rate']:.4f}\n")
                f.write(f"Rostros detectados: {affectnet_results['faces_detected']}\n\n")
                
                # Análisis detallado de imágenes seleccionadas
                f.write("ANÁLISIS DETALLADO (6 IMÁGENES REPRESENTATIVAS)\n")
                f.write("-" * 50 + "\n")
                for idx, result in enumerate(affectnet_results['detailed_results'], 1):
                    f.write(f"\nIMAGEN {idx}:\n")
                    f.write(f"  Archivo: {Path(result['image_path']).name}\n")
                    f.write(f"  Emoción real: {result['true_emotion']}\n")
                    
                    if result['original_pred']:
                        orig_pred = result['original_pred']
                        is_correct = orig_pred['predicted_class'] == result['true_class']
                        status = "CORRECTO" if is_correct else "INCORRECTO"
                        f.write(f"  Predicción imagen completa: {orig_pred['predicted_emotion']} ({orig_pred['confidence']:.3f}) - {status}\n")
                    
                    if result['face_detected'] and result['face_pred']:
                        face_pred = result['face_pred']
                        is_correct = face_pred['predicted_class'] == result['true_class']
                        status = "CORRECTO" if is_correct else "INCORRECTO"
                        f.write(f"  Predicción rostro detectado: {face_pred['predicted_emotion']} ({face_pred['confidence']:.3f}) - {status}\n")
                    else:
                        f.write(f"  Rostro detectado: No\n")
                
                # Comparación final
                f.write("\nCOMPARACIÓN IMAGEN COMPLETA vs ROSTRO DETECTADO\n")
                f.write("-" * 50 + "\n")
                f.write(f"Accuracy imagen completa: {affectnet_results['original_accuracy']:.3f}\n")
                f.write(f"Accuracy rostro detectado: {affectnet_results['face_accuracy']:.3f}\n")
                
                if affectnet_results['original_accuracy'] > affectnet_results['face_accuracy']:
                    f.write("CONCLUSIÓN: La imagen completa produce mejores resultados\n")
                elif affectnet_results['face_accuracy'] > affectnet_results['original_accuracy']:
                    f.write("CONCLUSIÓN: El rostro detectado produce mejores resultados\n")
                else:
                    f.write("CONCLUSIÓN: Ambos enfoques tienen rendimiento similar\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("=" * 80 + "\n")
        
        # Guardar resultados en JSON
        evaluation_data = {
            "timestamp": config_manager.timestamp,
            "model_path": str(model_path),
            "metrics": {
                "overall_accuracy": float(overall_accuracy),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
                "f1_per_class": {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))},
                "class_accuracies": {k: float(v) for k, v in class_accuracies.items()}
            },
            "affectnet_metrics": affectnet_results
        }
        
        results_json_path = results_dir / "evaluation_results.json"
        evaluation_data_clean = self.convert_numpy_types(evaluation_data)
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data_clean, f, indent=2, ensure_ascii=False)
        
        # Guardar métricas en CSV
        csv_path = self.save_evaluation_metrics_to_csv(results_dir, overall_accuracy, f1_weighted, f1_macro, f1_per_class, class_accuracies, class_names)
        
        return report_path, results_json_path
    
    def save_evaluation_metrics_to_csv(self, results_dir, overall_accuracy, f1_weighted, f1_macro, f1_per_class, class_accuracies, class_names, filename="evaluation_metrics.csv"):
        """Guarda las métricas de evaluación en un archivo CSV"""
        metrics_data = []
        
        # Métricas generales
        metrics_data.append({'metric_type': 'overall', 'metric_name': 'accuracy', 'value': overall_accuracy, 'class': 'all'})
        metrics_data.append({'metric_type': 'overall', 'metric_name': 'f1_weighted', 'value': f1_weighted, 'class': 'all'})
        metrics_data.append({'metric_type': 'overall', 'metric_name': 'f1_macro', 'value': f1_macro, 'class': 'all'})
        
        # Métricas por clase
        for i, class_name in enumerate(class_names):
            emotion_key = EMOTION_CLASSES[i]
            class_acc = class_accuracies.get(emotion_key, 0.0)
            class_f1 = f1_per_class[i]
            
            metrics_data.append({'metric_type': 'per_class', 'metric_name': 'accuracy', 'value': class_acc, 'class': class_name})
            metrics_data.append({'metric_type': 'per_class', 'metric_name': 'f1_score', 'value': class_f1, 'class': class_name})
        
        # Crear y guardar CSV
        df = pd.DataFrame(metrics_data)
        csv_path = results_dir / filename
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        print(f"Métricas de evaluación guardadas en: {csv_path}")
        return csv_path