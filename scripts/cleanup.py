#!/usr/bin/env python3
"""
Script de limpieza para Endurance Lab AI
Elimina archivos temporales y cache
"""
import os
import shutil
from pathlib import Path

# Configurar paths
PROJECT_ROOT = Path(__file__).parent.parent

def clean_pycache():
    """Elimina archivos __pycache__"""
    print("🧹 Limpiando archivos __pycache__...")
    
    cache_dirs = list(PROJECT_ROOT.rglob("__pycache__"))
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"  ✅ Eliminado: {cache_dir.relative_to(PROJECT_ROOT)}")
    
    print(f"🗑️ Eliminados {len(cache_dirs)} directorios de cache")

def clean_ds_store():
    """Elimina archivos .DS_Store (macOS)"""
    print("🧹 Limpiando archivos .DS_Store...")
    
    ds_files = list(PROJECT_ROOT.rglob(".DS_Store"))
    for ds_file in ds_files:
        ds_file.unlink()
        print(f"  ✅ Eliminado: {ds_file.relative_to(PROJECT_ROOT)}")
    
    print(f"🗑️ Eliminados {len(ds_files)} archivos .DS_Store")

def clean_temp_files():
    """Elimina archivos temporales"""
    print("🧹 Limpiando archivos temporales...")
    
    temp_patterns = ["*.tmp", "*.temp", "*.log", "*.bak"]
    temp_files = []
    
    for pattern in temp_patterns:
        temp_files.extend(PROJECT_ROOT.rglob(pattern))
    
    for temp_file in temp_files:
        if temp_file.is_file():
            temp_file.unlink()
            print(f"  ✅ Eliminado: {temp_file.relative_to(PROJECT_ROOT)}")
    
    print(f"🗑️ Eliminados {len(temp_files)} archivos temporales")

def clean_old_files():
    """Elimina archivos obsoletos de la reorganización"""
    print("🧹 Limpiando archivos obsoletos...")
    
    old_files = [
        "main_interface2.py",  # Archivo duplicado
    ]
    
    removed = 0
    for old_file in old_files:
        file_path = PROJECT_ROOT / old_file
        if file_path.exists():
            file_path.unlink()
            print(f"  ✅ Eliminado: {old_file}")
            removed += 1
    
    print(f"🗑️ Eliminados {removed} archivos obsoletos")

def clean_empty_dirs():
    """Elimina directorios vacíos"""
    print("🧹 Limpiando directorios vacíos...")
    
    removed = 0
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    print(f"  ✅ Eliminado directorio vacío: {dir_path.relative_to(PROJECT_ROOT)}")
                    removed += 1
            except OSError:
                pass  # Directorio no vacío o sin permisos
    
    print(f"🗑️ Eliminados {removed} directorios vacíos")

def show_cleanup_summary():
    """Muestra resumen del estado del proyecto"""
    print("\n📊 Resumen del proyecto:")
    
    # Contar archivos por tipo
    py_files = len(list(PROJECT_ROOT.rglob("*.py")))
    md_files = len(list(PROJECT_ROOT.rglob("*.md")))
    pdf_files = len(list(PROJECT_ROOT.rglob("*.pdf")))
    
    print(f"  📄 Archivos Python: {py_files}")
    print(f"  📝 Archivos Markdown: {md_files}")
    print(f"  📚 Archivos PDF: {pdf_files}")
    
    # Tamaño del proyecto
    total_size = sum(f.stat().st_size for f in PROJECT_ROOT.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"  💾 Tamaño total: {size_mb:.1f} MB")

def main():
    """Función principal de limpieza"""
    print("🧹 Endurance Lab AI - Limpieza del Proyecto")
    print("=" * 50)
    
    # Ejecutar limpiezas
    clean_pycache()
    clean_ds_store()
    clean_temp_files()
    clean_old_files()
    clean_empty_dirs()
    
    # Mostrar resumen
    show_cleanup_summary()
    
    print("\n✅ Limpieza completada exitosamente!")
    print("🚀 El proyecto está listo para uso o despliegue")

if __name__ == "__main__":
    main()