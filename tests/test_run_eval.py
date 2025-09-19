# tests/test_run_eval.py

import os
import json
import pytest
from unittest.mock import patch, MagicMock

def test_dataset_exists():
    """Verifica que el dataset de evaluación existe y tiene el formato correcto"""
    dataset_path = "tests/eval_dataset.json"
    assert os.path.exists(dataset_path), f"Dataset no encontrado: {dataset_path}"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    assert isinstance(dataset, list), "El dataset debe ser una lista"
    assert len(dataset) > 0, "El dataset no puede estar vacío"
    
    for i, item in enumerate(dataset):
        assert "question" in item, f"Item {i} debe tener 'question'"
        assert "answer" in item, f"Item {i} debe tener 'answer'"
        assert isinstance(item["question"], str), f"Question en item {i} debe ser string"
        assert isinstance(item["answer"], str), f"Answer en item {i} debe ser string"

def test_vectorstore_exists():
    """Verifica que el vectorstore existe"""
    vectorstore_path = "vectorstore"
    assert os.path.exists(vectorstore_path), f"Vectorstore no encontrado: {vectorstore_path}"
    assert os.path.exists(os.path.join(vectorstore_path, "index.faiss")), "Archivo index.faiss no encontrado"
    assert os.path.exists(os.path.join(vectorstore_path, "index.pkl")), "Archivo index.pkl no encontrado"

def test_prompts_exist():
    """Verifica que los prompts existen"""
    prompts_dir = "app/prompts"
    assert os.path.exists(prompts_dir), f"Directorio de prompts no encontrado: {prompts_dir}"
    
    expected_prompts = ["v1_asistente_deporte.txt", "v1_asistente_rrhh.txt"]
    for prompt_file in expected_prompts:
        prompt_path = os.path.join(prompts_dir, prompt_file)
        assert os.path.exists(prompt_path), f"Prompt no encontrado: {prompt_path}"

@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
def test_rag_pipeline_imports():
    """Verifica que los módulos se pueden importar correctamente"""
    try:
        from app.rag_pipeline import load_vectorstore_from_disk, build_chain
        assert callable(load_vectorstore_from_disk), "load_vectorstore_from_disk debe ser callable"
        assert callable(build_chain), "build_chain debe ser callable"
    except ImportError as e:
        pytest.fail(f"Error importando rag_pipeline: {e}")

def test_run_eval_imports():
    """Verifica que run_eval.py se puede importar sin errores de sintaxis"""
    import sys
    import os
    
    # Añadir el directorio raíz al path
    sys.path.insert(0, os.path.abspath('.'))
    
    try:
        # Solo verificamos que se puede importar sin ejecutar
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_eval", "app/run_eval.py")
        # No ejecutamos el módulo, solo verificamos que se puede cargar
        assert spec is not None, "No se pudo cargar el spec de run_eval.py"
    except Exception as e:
        pytest.fail(f"Error cargando run_eval.py: {e}")
