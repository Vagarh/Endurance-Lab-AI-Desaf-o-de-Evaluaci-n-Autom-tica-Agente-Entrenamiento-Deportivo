#!/usr/bin/env python3
"""
Script de verificación del setup del proyecto
Verifica que todos los archivos necesarios existen antes de ejecutar tests
"""

import os
import sys
import json

def check_file_exists(filepath, description):
    """Verifica que un archivo existe"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} no encontrado: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Verifica que un directorio existe"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} no encontrado: {dirpath}")
        return False

def verify_dataset_format(filepath):
    """Verifica el formato del dataset de evaluación"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not isinstance(dataset, list):
            print(f"❌ Dataset debe ser una lista: {filepath}")
            return False
        
        if len(dataset) == 0:
            print(f"❌ Dataset está vacío: {filepath}")
            return False
        
        for i, item in enumerate(dataset):
            if not isinstance(item, dict):
                print(f"❌ Item {i} debe ser un diccionario")
                return False
            if "question" not in item or "answer" not in item:
                print(f"❌ Item {i} debe tener 'question' y 'answer'")
                return False
        
        print(f"✅ Dataset válido con {len(dataset)} items: {filepath}")
        return True
    
    except Exception as e:
        print(f"❌ Error verificando dataset {filepath}: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🔍 Verificando setup del proyecto...")
    
    all_good = True
    
    # Archivos críticos
    critical_files = [
        ("requirements.txt", "Archivo de dependencias"),
        ("app/rag_pipeline.py", "Módulo RAG pipeline"),
        ("app/run_eval.py", "Script de evaluación"),
        ("tests/test_run_eval.py", "Tests de evaluación"),
    ]
    
    for filepath, description in critical_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Directorios críticos
    critical_dirs = [
        ("app/prompts", "Directorio de prompts"),
        ("vectorstore", "Directorio de vectorstore"),
        ("tests", "Directorio de tests"),
    ]
    
    for dirpath, description in critical_dirs:
        if not check_directory_exists(dirpath, description):
            all_good = False
    
    # Dataset de evaluación
    dataset_path = "tests/eval_dataset.json"
    if check_file_exists(dataset_path, "Dataset de evaluación"):
        if not verify_dataset_format(dataset_path):
            all_good = False
    else:
        all_good = False
    
    # Verificar archivos de vectorstore
    vectorstore_files = ["vectorstore/index.faiss", "vectorstore/index.pkl"]
    for vf in vectorstore_files:
        if not check_file_exists(vf, f"Archivo vectorstore"):
            all_good = False
    
    # Verificar prompts
    prompt_files = [
        "app/prompts/v1_asistente_deporte.txt",
        "app/prompts/v1_asistente_rrhh.txt"
    ]
    for pf in prompt_files:
        if not check_file_exists(pf, f"Archivo de prompt"):
            all_good = False
    
    if all_good:
        print("\n🎉 ¡Verificación completada exitosamente!")
        print("✅ Todos los archivos necesarios están presentes")
        return 0
    else:
        print("\n❌ Verificación falló")
        print("🔧 Algunos archivos necesarios no están presentes")
        return 1

if __name__ == "__main__":
    sys.exit(main())