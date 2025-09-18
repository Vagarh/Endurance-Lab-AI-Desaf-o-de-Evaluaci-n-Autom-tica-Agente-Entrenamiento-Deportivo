#!/usr/bin/env python3
"""
Health check script for Endurance Lab AI
Verifica que todos los componentes estén funcionando correctamente
"""

import sys
import os
import requests
import time
from pathlib import Path

def check_streamlit_health():
    """Verifica que Streamlit esté respondiendo"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_nginx_health():
    """Verifica que Nginx esté respondiendo"""
    try:
        response = requests.get("http://localhost/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_vectorstore():
    """Verifica que el vectorstore existe"""
    vectorstore_path = Path("vectorstore")
    return vectorstore_path.exists() and any(vectorstore_path.iterdir())

def check_env_vars():
    """Verifica variables de entorno críticas"""
    required_vars = ["OPENAI_API_KEY"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    return len(missing) == 0, missing

def main():
    print("🔍 Verificando estado de Endurance Lab AI...")
    
    all_good = True
    
    # Check environment variables
    env_ok, missing_vars = check_env_vars()
    if env_ok:
        print("✅ Variables de entorno: OK")
    else:
        print(f"❌ Variables de entorno faltantes: {missing_vars}")
        all_good = False
    
    # Check vectorstore
    if check_vectorstore():
        print("✅ Vectorstore: OK")
    else:
        print("❌ Vectorstore: No encontrado")
        all_good = False
    
    # Check Streamlit
    print("🔄 Verificando Streamlit...")
    for i in range(3):
        if check_streamlit_health():
            print("✅ Streamlit: OK")
            break
        else:
            if i < 2:
                print(f"⏳ Streamlit no responde, reintentando... ({i+1}/3)")
                time.sleep(5)
            else:
                print("❌ Streamlit: No responde")
                all_good = False
    
    # Check Nginx
    if check_nginx_health():
        print("✅ Nginx: OK")
    else:
        print("⚠️ Nginx: No responde (opcional)")
    
    if all_good:
        print("\n🎉 ¡Todos los componentes están funcionando correctamente!")
        print("📱 Aplicación disponible en: http://localhost:8501")
        return 0
    else:
        print("\n❌ Algunos componentes tienen problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main())