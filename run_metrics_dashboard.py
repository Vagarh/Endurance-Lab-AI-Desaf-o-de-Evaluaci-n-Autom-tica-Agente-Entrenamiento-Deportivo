#!/usr/bin/env python3
"""
Script para ejecutar el Dashboard de Métricas de Endurance Lab AI
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 Iniciando Dashboard de Métricas - Endurance Lab AI")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not Path("app/metrics_dashboard.py").exists():
        print("❌ Error: No se encuentra app/metrics_dashboard.py")
        print("   Asegúrate de ejecutar este script desde la raíz del proyecto")
        return 1
    
    # Verificar que Streamlit está instalado
    try:
        import streamlit
        print(f"✅ Streamlit encontrado: v{streamlit.__version__}")
    except ImportError:
        print("❌ Error: Streamlit no está instalado")
        print("   Ejecuta: pip install -r requirements.txt")
        return 1
    
    # Verificar dependencias adicionales
    try:
        import plotly
        print(f"✅ Plotly encontrado: v{plotly.__version__}")
    except ImportError:
        print("❌ Error: Plotly no está instalado")
        print("   Ejecuta: pip install plotly kaleido")
        return 1
    
    print("\n📊 Configuración del Dashboard:")
    print("   - Puerto: 8502")
    print("   - URL: http://localhost:8502")
    print("   - Archivo: app/metrics_dashboard.py")
    
    print("\n🔄 Iniciando servidor...")
    
    try:
        # Ejecutar Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "app/metrics_dashboard.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--server.headless=false"
        ]
        
        print(f"   Comando: {' '.join(cmd)}")
        print("\n" + "=" * 60)
        print("🌐 Dashboard iniciado exitosamente!")
        print("📱 Abre tu navegador en: http://localhost:8502")
        print("⏹️  Presiona Ctrl+C para detener el servidor")
        print("=" * 60)
        
        # Abrir navegador automáticamente después de un momento
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open("http://localhost:8502")
                print("🌐 Navegador abierto automáticamente")
            except:
                pass
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Ejecutar Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Dashboard detenido por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error ejecutando dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())