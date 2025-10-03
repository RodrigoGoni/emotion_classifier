"""
Arquitectura CNN personalizada para clasificación de emociones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=7, input_size=(100, 100), num_channels=3, dropout_prob=0.5, stride=1):
        super(CNNModel, self).__init__()
        
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calcular tamaño dinámicamente
        conv_output_size = self._get_conv_output_size()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )
    
    def _get_conv_output_size(self):
        """Calcula el tamaño de salida de las capas convolucionales"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            dummy_output = self.features(dummy_input)
            return dummy_output.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.classifier(x)
        return x