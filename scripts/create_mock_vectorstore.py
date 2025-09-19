#!/usr/bin/env python3
"""
Script para crear un vectorstore mock para tests cuando no hay API key válida
"""

import os
import sys
import pickle
import numpy as np

def create_mock_vectorstore():
    """Crea un vectorstore mock para tests"""
    
    vectorstore_dir = "vectorstore"
    os.makedirs(vectorstore_dir, exist_ok=True)
    
    # Crear un archivo index.faiss mock (vacío pero válido)
    faiss_path = os.path.join(vectorstore_dir, "index.faiss")
    with open(faiss_path, "wb") as f:
        # Escribir algunos bytes para simular un archivo FAISS
        f.write(b"MOCK_FAISS_INDEX")
    
    # Crear un archivo index.pkl mock
    pkl_path = os.path.join(vectorstore_dir, "index.pkl")
    mock_data = {
        "index_to_docstore_id": {},
        "docstore": {},
        "index": "mock_index"
    }
    
    with open(pkl_path, "wb") as f:
        pickle.dump(mock_data, f)
    
    print(f"✅ Vectorstore mock creado en {vectorstore_dir}")
    print(f"📁 Archivos: {os.listdir(vectorstore_dir)}")
    
    return True

if __name__ == "__main__":
    create_mock_vectorstore()