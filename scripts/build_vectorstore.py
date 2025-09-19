#!/usr/bin/env python3
"""
Script para generar el vectorstore desde los PDFs
"""

import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath('.'))

def main():
    """Genera el vectorstore desde los PDFs"""
    
    # Verificar que tenemos API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY no configurada")
        print("💡 Configura tu API key en las variables de entorno")
        return 1
    
    # Verificar que existen PDFs
    pdf_dir = "data/pdfs"
    if not os.path.exists(pdf_dir):
        print(f"❌ Error: Directorio de PDFs no encontrado: {pdf_dir}")
        return 1
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ Error: No se encontraron archivos PDF en {pdf_dir}")
        return 1
    
    print(f"📚 Encontrados {len(pdf_files)} archivos PDF")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    try:
        # Importar y ejecutar
        from app.rag_pipeline import save_vectorstore
        
        print("🔄 Generando vectorstore...")
        save_vectorstore()
        
        # Verificar que se generó correctamente
        vectorstore_dir = "vectorstore"
        if os.path.exists(vectorstore_dir):
            files = os.listdir(vectorstore_dir)
            print(f"✅ Vectorstore generado exitosamente en {vectorstore_dir}")
            print(f"📁 Archivos generados: {files}")
        else:
            print("❌ Error: Vectorstore no se generó correctamente")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"❌ Error generando vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())