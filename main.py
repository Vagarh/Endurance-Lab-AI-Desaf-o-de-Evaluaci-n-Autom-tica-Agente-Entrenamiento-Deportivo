#!/usr/bin/env python3
"""
Endurance Lab AI - Punto de entrada principal
Versión 2.0.0 con estructura organizada
"""
import sys
import os
from pathlib import Path

# Agregar directorios al path
PROJECT_ROOT = Path(__file__).parent
sys.path.extend([
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "app"),
    str(PROJECT_ROOT / "utils"),
    str(PROJECT_ROOT / "scripts")
])

def main():
    """Función principal de entrada"""
    print("🏆 Endurance Lab AI v2.0.0")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            from scripts.start import main as start_main
            start_main()
            
        elif command == "setup":
            from scripts.setup_project import main as setup_main
            setup_main()
            
        elif command == "test":
            from scripts.test_basic import main as test_main
            test_main()
            
        elif command == "create-vectorstore":
            from scripts.create_vectorstore import main as create_main
            create_main()
            
        elif command == "config":
            from utils.config import get_config_summary, validate_config
            print("📋 Configuración actual:")
            config = get_config_summary()
            for section, data in config.items():
                print(f"\n{section.upper()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            
            print("\n🔍 Validación:")
            issues = validate_config()
            if issues:
                for issue in issues:
                    print(f"  ❌ {issue}")
            else:
                print("  ✅ Configuración válida")
                
        elif command == "help":
            show_help()
            
        else:
            print(f"❌ Comando desconocido: {command}")
            show_help()
    else:
        show_help()

def show_help():
    """Muestra la ayuda de comandos"""
    print("""
🎯 Comandos disponibles:

📋 CONFIGURACIÓN:
  python main.py setup          - Configuración inicial del proyecto
  python main.py config         - Mostrar configuración actual
  python main.py test           - Ejecutar tests básicos

🚀 EJECUCIÓN:
  python main.py start          - Iniciar la aplicación
  python main.py create-vectorstore - Crear base de datos vectorial

📚 AYUDA:
  python main.py help           - Mostrar esta ayuda

🌐 ACCESO DIRECTO:
  streamlit run app/main_interface_improved.py

📁 ESTRUCTURA DEL PROYECTO:
  app/          - Aplicación principal
  scripts/      - Scripts de utilidad
  utils/        - Configuración y utilidades
  data/         - Datos de entrenamiento
  assets/       - Recursos estáticos
  docs/         - Documentación
  tests/        - Tests y evaluaciones
    """)

if __name__ == "__main__":
    main()