"""
Dataset personalizado para clasificación de emociones
"""
from collections import Counter
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class EmotionDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self._stats = None
        
    def get_class_counts(self):
        """Retorna el número de imágenes por clase"""
        if self._stats is None:
            self._compute_stats()
        return self._stats['class_counts']
    
    def get_num_classes(self):
        """Retorna el número total de clases"""
        return len(self.classes)
    
    def get_total_images(self):
        """Retorna el número total de imágenes"""
        return len(self.samples)
    
    def get_image_dimensions(self):
        """Retorna dimensiones promedio de las imágenes"""
        if self._stats is None:
            self._compute_stats()
        return self._stats['dimensions']
    
    def _compute_stats(self):
        """Calcula estadísticas básicas del dataset"""
        class_counts = Counter()
        widths, heights = [], []
        
        for img_path, class_idx in self.samples:
            class_name = self.classes[class_idx]
            class_counts[class_name] += 1
        
        sample_paths = [self.samples[i][0] for i in range(0, len(self.samples), 50)]
        
        for path in sample_paths:
            try:
                with Image.open(path) as img:
                    widths.append(img.size[0])
                    heights.append(img.size[1])
            except:
                continue
        
        avg_width = sum(widths) / len(widths) if widths else 0
        avg_height = sum(heights) / len(heights) if heights else 0
        
        self._stats = {
            'class_counts': dict(class_counts),
            'dimensions': {
                'avg_width': int(avg_width),
                'avg_height': int(avg_height),
                'samples_analyzed': len(widths)
            }
        }
    
    def print_info(self):
        """Imprime información del dataset"""
        print(f"Dataset Information:")
        print(f"Total images: {self.get_total_images()}")
        print(f"Number of classes: {self.get_num_classes()}")
        print(f"Classes: {self.classes}")
        
        print(f"\nImages per class:")
        for class_name, count in self.get_class_counts().items():
            print(f"  {class_name}: {count}")
        
        dims = self.get_image_dimensions()
        print(f"\nImage dimensions (average):")
        print(f"  Width: {dims['avg_width']}px")
        print(f"  Height: {dims['avg_height']}px")
        print(f"  Samples analyzed: {dims['samples_analyzed']}")
        
    def plot_histogram(self):
        """Genera un histograma de la distribución de clases"""
        
        class_counts = self.get_class_counts()
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Crear directorio results/plots
        plot_path = Path('results/plots')
        plot_path.mkdir(parents=True, exist_ok=True)
        figure_path = plot_path / 'class_distribution.png'
        
        plt.figure(figsize=(10, 6))
        plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Clases')
        plt.ylabel('Número de imágenes')
        plt.title('Distribución de clases en el dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()
        
        print(f"Histograma guardado en: {figure_path}")
        
    def plot_sample_images(self, num_images=9):
        """Muestra algunas imágenes de ejemplo del dataset"""
        if num_images <= 0:
            print("Número de imágenes debe ser mayor que 0")
            return
        
        num_images = min(num_images, len(self.samples))
        cols = int(num_images**0.5)
        rows = (num_images + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for i in range(num_images):
            img_path, class_idx = self.samples[i]
            class_name = self.classes[class_idx]
            
            try:
                with Image.open(img_path) as img:
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img)
                    plt.title(class_name)
                    plt.axis('off')
            except:
                continue
        
        plt.tight_layout()
        
        # Crear directorio results/plots
        plot_path = Path('results/plots')
        plot_path.mkdir(parents=True, exist_ok=True)
        figure_path = plot_path / 'sample_images.png'
        
        plt.savefig(figure_path)
        plt.close()
        
        print(f"Imágenes de muestra guardadas en: {figure_path}")