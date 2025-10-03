"""
Transformaciones básicas para el preprocesamiento de imágenes
"""
from torchvision.transforms import  v2 as transforms
import torch

def get_train_transforms():
    """Transformaciones para entrenamiento - optimizadas para emociones faciales"""
    return transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomRotation(degrees=10),  # Rotación suave para no distorsionar expresiones
        transforms.RandomHorizontalFlip(p=0.3),  # Flip reducido - algunas emociones son asimétricas
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variaciones de iluminación
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Pequeños desplazamientos
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])



def get_val_transforms():
    """Transformaciones para validación"""
    return transforms.Compose([
        transforms.Resize((100, 100)),  # Mismo tamaño que entrenamiento
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])