#!/usr/bin/env python3
"""
Script para reiniciar la aplicación con la interfaz mejorada
"""
import subprocess
import sys
import time
import os
from pathlib import Path

def stop_streamlit():
    """Intenta detener procesos de Streamlit existentes"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], 
                         capture_output=True, check=False)
        else:  # Unix/Linux/Mac
            subprocess.run(['pkill', '-f', 'streamlit'], 
                         capture_output=True, check=False)
        print("🛑 Procesos de Streamlit detenidos")
    except Exception as e:
        print(f"⚠️ No se pudieron detener procesos existentes: {e}")

def start_improved_app():
    """Inicia la aplicación con la interfaz mejorada"""
    print("🚀 Iniciando Endurance Lab AI con interfaz mejorada...")
    
    # Verificar que existe la versión mejorada
    improved_app = Path("app/main_interface_improved.py")
    if not improved_app.exists():
        print("❌ No se encontró la interfaz mejorada")
        return False
    
    try:
        # Iniciar Streamlit con la interfaz mejorada
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(improved_app),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--theme.base", "light",
            "--theme.primaryColor", "#2E86AB",
            "--theme.backgroundColor", "#FFFFFF",
            "--theme.secondaryBackgroundColor", "#F0F2F6"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error iniciando la aplicación: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
        return True
    
    return True

def main():
    print("🔄 Reiniciando Endurance Lab AI con interfaz mejorada...\n")
    
    # Detener procesos existentes
    stop_streamlit()
    
    # Esperar un momento
    time.sleep(2)
    
    # Iniciar la aplicación mejorada
    print("🌐 La aplicación se abrirá en: http://localhost:8501")
    print("⏹️  Presiona Ctrl+C para detener\n")
    
    success = start_improved_app()
    
    if success:
        print("\n✅ Aplicación ejecutada correctamente")
    else:
        print("\n❌ Error al ejecutar la aplicación")

if __name__ == "__main__":
    main()