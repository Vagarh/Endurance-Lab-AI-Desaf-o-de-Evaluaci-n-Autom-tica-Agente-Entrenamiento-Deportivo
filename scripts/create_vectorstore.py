#!/usr/bin/env python3
"""
Script para crear el vectorstore desde los PDFs
"""
import os
from dotenv import load_dotenv
from app.rag_pipeline import save_vectorstore

load_dotenv()

if __name__ == "__main__":
    print("🔄 Creando vectorstore desde los PDFs...")
    
    # Verificar que existe la carpeta de PDFs
    if not os.path.exists("data/pdfs"):
        print("❌ Error: No existe la carpeta data/pdfs")
        exit(1)
    
    # Verificar que hay PDFs
    pdf_files = [f for f in os.listdir("data/pdfs") if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ Error: No se encontraron archivos PDF en data/pdfs")
        exit(1)
    
    print(f"📚 Encontrados {len(pdf_files)} archivos PDF:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    try:
        save_vectorstore()
        print("✅ Vectorstore creado exitosamente en ./vectorstore")
    except Exception as e:
        print(f"❌ Error al crear vectorstore: {e}")
        exit(1)