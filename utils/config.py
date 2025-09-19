"""
Configuración centralizada para Endurance Lab AI
Actualizada para la nueva estructura de carpetas
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Directorios del proyecto (nueva estructura)
PROJECT_ROOT = Path(__file__).parent.parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data" / "pdfs"
PROMPT_DIR = APP_DIR / "prompts"
VECTOR_DIR = PROJECT_ROOT / "vectorstore"
TESTS_DIR = PROJECT_ROOT / "tests"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Configuración de RAG
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_deporte")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Configuración de evaluación
DATASET_PATH = os.getenv("DATASET_PATH", "tests/eval_dataset.json")

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Configuración de Streamlit
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Información del proyecto
PROJECT_NAME = "Endurance Lab AI"
PROJECT_VERSION = "2.0.0"  # Actualizada por la reorganización
PROJECT_DESCRIPTION = "Asistente virtual especializado en entrenamiento de resistencia"

# URLs y recursos (nueva estructura)
REPO_URL = "https://github.com/Vagarh/Endurance-Lab-AI-Desaf-o-de-Evaluaci-n-Autom-tica-Agente-Entrenamiento-Deportivo"
LOGO_PATH = ASSETS_DIR / "images" / "ChatGPT Image 2 may 2025, 17_42_01.png"
HERO_PATH = ASSETS_DIR / "images" / "ChatGPT Image 2 may 2025, 17_45_41.png"

# Disciplinas deportivas soportadas
SUPPORTED_SPORTS = [
    "Ciclismo",
    "Running", 
    "Triatlón",
    "Natación",
    "Otro"
]

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def validate_config():
    """Valida la configuración del proyecto"""
    issues = []
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "tu_clave_aqui":
        issues.append("OPENAI_API_KEY no configurada")
    
    if not DATA_DIR.exists():
        issues.append(f"Directorio de datos no existe: {DATA_DIR}")
    
    if not PROMPT_DIR.exists():
        issues.append(f"Directorio de prompts no existe: {PROMPT_DIR}")
    
    if not APP_DIR.exists():
        issues.append(f"Directorio de aplicación no existe: {APP_DIR}")
    
    # Verificar que existe al menos un PDF
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if len(pdf_files) == 0:
        issues.append("No se encontraron archivos PDF en el directorio de datos")
    
    return issues

def get_config_summary():
    """Retorna un resumen de la configuración actual"""
    return {
        "project": {
            "name": PROJECT_NAME,
            "version": PROJECT_VERSION,
            "root": str(PROJECT_ROOT)
        },
        "openai": {
            "model": OPENAI_MODEL,
            "temperature": TEMPERATURE,
            "api_key_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "tu_clave_aqui")
        },
        "rag": {
            "prompt_version": PROMPT_VERSION,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        },
        "paths": {
            "app_dir": str(APP_DIR),
            "data_dir": str(DATA_DIR),
            "vector_dir": str(VECTOR_DIR),
            "prompt_dir": str(PROMPT_DIR),
            "assets_dir": str(ASSETS_DIR),
            "docs_dir": str(DOCS_DIR)
        },
        "structure": {
            "organized": True,
            "version": "2.0.0"
        }
    }

def get_project_structure():
    """Retorna la estructura del proyecto"""
    return {
        "root": PROJECT_ROOT,
        "directories": {
            "app": "Aplicación principal y módulos",
            "data": "Datos de entrenamiento (PDFs)",
            "tests": "Tests y datasets de evaluación", 
            "scripts": "Scripts de utilidad y automatización",
            "utils": "Utilidades y configuración",
            "assets": "Recursos estáticos (imágenes, docs)",
            "docs": "Documentación del proyecto",
            "vectorstore": "Base de datos vectorial",
            "mlruns": "Experimentos de MLflow"
        }
    }