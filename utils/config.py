"""
Configuración centralizada para Endurance Lab AI
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "pdfs"
PROMPT_DIR = PROJECT_ROOT / "app" / "prompts"
VECTOR_DIR = PROJECT_ROOT / "vectorstore"
TESTS_DIR = PROJECT_ROOT / "tests"

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
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Asistente virtual especializado en entrenamiento de resistencia"

# URLs y recursos
REPO_URL = "https://github.com/Vagarh/Endurance-Lab-AI-Desaf-o-de-Evaluaci-n-Autom-tica-Agente-Entrenamiento-Deportivo"
LOGO_PATH = PROJECT_ROOT / "Imagenes" / "ChatGPT Image 2 may 2025, 17_42_01.png"
HERO_PATH = PROJECT_ROOT / "Imagenes" / "ChatGPT Image 2 may 2025, 17_45_41.png"

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
            "data_dir": str(DATA_DIR),
            "vector_dir": str(VECTOR_DIR),
            "prompt_dir": str(PROMPT_DIR)
        }
    }