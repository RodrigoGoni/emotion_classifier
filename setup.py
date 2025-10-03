"""
Setup script para el proyecto de clasificación de emociones
"""
import subprocess
import sys
import os

def install_requirements():
    """Instala las dependencias desde requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False

def verify_installation():
    """Verifica que las librerías principales estén instaladas"""
    packages = ["torch", "torchvision", "opencv-python", "numpy", "matplotlib", "scikit-learn", "pandas"]
    missing = []
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def main():
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt no encontrado")
        return
    
    print("Instalando dependencias...")
    if install_requirements():
        print("Dependencias instaladas")
        
        success, missing = verify_installation()
        if success:
            print("Verificación exitosa")
        else:
            print(f"Paquetes faltantes: {missing}")
    else:
        print("Error en instalación")

if __name__ == "__main__":
    main()