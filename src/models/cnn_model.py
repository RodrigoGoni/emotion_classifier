"""
Arquitectura CNN personalizada para clasificación de emociones
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=7, input_size=(100, 100), num_channels=3, dropout_prob=0.5, 
                 conv_layers=None, fc_layers=None, stride=1):
        super(CNNModel, self).__init__()
        
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        
        # Configuración por defecto si no se proporciona
        if conv_layers is None:
            conv_layers = [
                {'filters': 64, 'kernel': 3, 'pool': 2},
                {'filters': 128, 'kernel': 3, 'pool': 2},
                {'filters': 256, 'kernel': 3, 'pool': 2},
                {'filters': 512, 'kernel': 3, 'pool': 2}
            ]
        
        if fc_layers is None:
            fc_layers = [1024, 512, 256, 32]
        
        self.conv_layers_config = conv_layers
        self.fc_layers_config = fc_layers
        
        # Construir capas convolucionales dinámicamente
        features_layers = []
        in_channels = num_channels
        
        for i, layer_config in enumerate(conv_layers):
            out_channels = layer_config['filters']
            kernel_size = layer_config['kernel']
            pool_size = layer_config['pool']
            
            # Capa convolucional
            features_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2, stride=stride)
            )
            features_layers.append(nn.ReLU(inplace=True))
            
            # Pooling si se especifica
            if pool_size > 1:
                features_layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*features_layers)
        
        # Calcular tamaño dinámicamente
        conv_output_size = self._get_conv_output_size()
        
        # Construir clasificador dinámicamente
        classifier_layers = []
        
        # Primera capa (desde conv output al primer FC)
        if fc_layers:
            classifier_layers.append(nn.Dropout(p=dropout_prob))
            classifier_layers.append(nn.Linear(conv_output_size, fc_layers[0]))
            classifier_layers.append(nn.ReLU(inplace=True))
            
            # Capas intermedias
            for i in range(len(fc_layers) - 1):
                classifier_layers.append(nn.Dropout(p=dropout_prob))
                classifier_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
                classifier_layers.append(nn.ReLU(inplace=True))
            
            # Capa final
            classifier_layers.append(nn.Dropout(p=dropout_prob))
            classifier_layers.append(nn.Linear(fc_layers[-1], num_classes))
        else:
            # Fallback: conexión directa
            classifier_layers.append(nn.Dropout(p=dropout_prob))
            classifier_layers.append(nn.Linear(conv_output_size, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
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
    
    def get_model_info(self):
        """Retorna información sobre la arquitectura del modelo"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'conv_layers': self.conv_layers_config,
            'fc_layers': self.fc_layers_config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'dropout_prob': self.dropout_prob
        }