#!/usr/bin/env python3
"""
Script de inicio rápido para Endurance Lab AI
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Verifica que las dependencias estén instaladas"""
    try:
        import streamlit
        import langchain
        import mlflow
        return True
    except ImportError as e:
        print(f"❌ Dependencias faltantes: {e}")
        print("📦 Ejecuta: pip install -r requirements.txt")
        return False

def check_api_key():
    """Verifica la configuración de la API key"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'tu_clave_aqui':
        print("⚠️  OPENAI_API_KEY no configurada")
        print("📝 Edita el archivo .env y agrega tu clave de OpenAI")
        return False
    return True

def check_vectorstore():
    """Verifica si existe el vectorstore"""
    return Path("vectorstore").exists()

def create_vectorstore():
    """Crea el vectorstore"""
    print("🔄 Creando vectorstore...")
    try:
        from app.rag_pipeline import save_vectorstore
        save_vectorstore()
        print("✅ Vectorstore creado")
        return True
    except Exception as e:
        print(f"❌ Error creando vectorstore: {e}")
        return False

def start_streamlit():
    """Inicia la aplicación Streamlit"""
    print("🚀 Iniciando Endurance Lab AI...")
    
    # Usar la versión mejorada si existe, sino la original
    if Path("app/main_interface_improved.py").exists():
        app_file = "app/main_interface_improved.py"
    else:
        app_file = "app/main_interface.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error iniciando Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
        return True

def main():
    print("🏆 Endurance Lab AI - Inicio Rápido\n")
    
    # Verificar dependencias
    if not check_requirements():
        return
    
    # Verificar API key
    if not check_api_key():
        print("\n💡 Pasos para configurar:")
        print("   1. Copia tu API key de OpenAI")
        print("   2. Edita el archivo .env")
        print("   3. Reemplaza 'tu_clave_aqui' con tu API key")
        print("   4. Ejecuta este script nuevamente")
        return
    
    # Verificar/crear vectorstore
    if not check_vectorstore():
        print("📚 Vectorstore no encontrado, creando...")
        if not create_vectorstore():
            return
    else:
        print("✅ Vectorstore encontrado")
    
    # Iniciar aplicación
    print("\n🎯 Todo listo! Iniciando aplicación...")
    print("🌐 La aplicación se abrirá en: http://localhost:8501")
    print("⏹️  Presiona Ctrl+C para detener\n")
    
    start_streamlit()

if __name__ == "__main__":
    main()