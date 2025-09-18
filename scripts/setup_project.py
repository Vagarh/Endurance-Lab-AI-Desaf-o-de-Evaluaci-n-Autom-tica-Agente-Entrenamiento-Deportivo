#!/usr/bin/env python3
"""
Script de configuración inicial para Endurance Lab AI
"""
import os
import sys
from pathlib import Path

def check_api_key():
    """Verifica si existe una API key de OpenAI configurada"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'tu_clave_aqui':
        print("⚠️  OPENAI_API_KEY no configurada")
        print("📝 Edita el archivo .env y agrega tu clave de OpenAI")
        return False
    
    print("✅ OPENAI_API_KEY configurada")
    return True

def create_vectorstore():
    """Crea el vectorstore si no existe"""
    if os.path.exists("vectorstore"):
        print("✅ Vectorstore ya existe")
        return True
    
    print("🔄 Creando vectorstore...")
    try:
        from app.rag_pipeline import save_vectorstore
        save_vectorstore()
        print("✅ Vectorstore creado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error creando vectorstore: {e}")
        return False

def run_basic_tests():
    """Ejecuta tests básicos del sistema"""
    print("🧪 Ejecutando tests básicos...")
    
    try:
        # Test de importaciones
        import streamlit
        import langchain
        import mlflow
        from app.rag_pipeline import load_documents
        
        # Test de archivos
        required_files = [
            "app/main_interface.py",
            "app/rag_pipeline.py",
            "app/prompts/v1_asistente_deporte.txt"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Archivo faltante: {file_path}")
                return False
        
        # Test de PDFs
        pdf_files = list(Path("data/pdfs").glob("*.pdf"))
        if len(pdf_files) == 0:
            print("❌ No se encontraron archivos PDF en data/pdfs")
            return False
        
        print(f"✅ Tests básicos completados ({len(pdf_files)} PDFs encontrados)")
        return True
        
    except ImportError as e:
        print(f"❌ Error en importaciones: {e}")
        return False

def main():
    print("🚀 Configurando Endurance Lab AI...\n")
    
    # Verificar estructura básica
    if not run_basic_tests():
        print("❌ Tests básicos fallaron")
        sys.exit(1)
    
    # Verificar API key
    has_api_key = check_api_key()
    
    if has_api_key:
        # Crear vectorstore si es necesario
        if create_vectorstore():
            print("\n🎉 ¡Proyecto configurado correctamente!")
            print("\n📋 Comandos disponibles:")
            print("   • Interfaz web: streamlit run app/main_interface.py")
            print("   • Evaluación: python app/run_eval.py")
            print("   • Tests: python test_basic.py")
        else:
            print("\n⚠️  Configuración parcial completada")
            print("❌ Error al crear vectorstore - verifica tu API key")
    else:
        print("\n⚠️  Configuración parcial completada")
        print("📝 Configura tu OPENAI_API_KEY en .env para continuar")
    
    print(f"\n📁 Directorio del proyecto: {os.getcwd()}")

if __name__ == "__main__":
    main()