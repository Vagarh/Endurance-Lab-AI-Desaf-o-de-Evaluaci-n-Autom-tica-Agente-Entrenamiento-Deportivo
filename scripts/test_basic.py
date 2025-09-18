#!/usr/bin/env python3
"""
Test básico para verificar que las importaciones funcionan
"""
import sys
import os

def test_imports():
    print("🧪 Probando importaciones básicas...")
    
    try:
        import streamlit as st
        print("✅ Streamlit importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Streamlit: {e}")
        return False
    
    try:
        import langchain
        print("✅ LangChain importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando LangChain: {e}")
        return False
    
    try:
        import mlflow
        print("✅ MLflow importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando MLflow: {e}")
        return False
    
    try:
        from app.rag_pipeline import load_documents
        print("✅ Módulo RAG pipeline importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando RAG pipeline: {e}")
        return False
    
    return True

def test_file_structure():
    print("\n📁 Verificando estructura de archivos...")
    
    required_files = [
        "app/main_interface.py",
        "app/rag_pipeline.py", 
        "app/run_eval.py",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "app/prompts",
        "data/pdfs",
        "tests"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            all_good = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NO ENCONTRADO")
            all_good = False
    
    return all_good

def test_pdfs():
    print("\n📚 Verificando archivos PDF...")
    
    pdf_dir = "data/pdfs"
    if not os.path.exists(pdf_dir):
        print(f"❌ Directorio {pdf_dir} no existe")
        return False
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"📄 Encontrados {len(pdf_files)} archivos PDF:")
    
    for pdf in pdf_files[:5]:  # Mostrar solo los primeros 5
        print(f"  - {pdf}")
    
    if len(pdf_files) > 5:
        print(f"  ... y {len(pdf_files) - 5} más")
    
    return len(pdf_files) > 0

if __name__ == "__main__":
    print("🚀 Ejecutando tests básicos del proyecto Endurance Lab AI\n")
    
    success = True
    success &= test_imports()
    success &= test_file_structure() 
    success &= test_pdfs()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 ¡Todos los tests básicos pasaron!")
        print("📝 Próximos pasos:")
        print("   1. Agregar tu OPENAI_API_KEY al archivo .env")
        print("   2. Ejecutar: python create_vectorstore.py")
        print("   3. Ejecutar: streamlit run app/main_interface.py")
    else:
        print("❌ Algunos tests fallaron. Revisa los errores arriba.")
    
    sys.exit(0 if success else 1)