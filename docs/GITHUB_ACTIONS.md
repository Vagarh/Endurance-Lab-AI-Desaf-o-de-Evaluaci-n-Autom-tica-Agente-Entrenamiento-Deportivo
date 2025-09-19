# GitHub Actions - Configuración y Uso

## ¿Qué son GitHub Actions?

GitHub Actions es un sistema de CI/CD (Integración Continua/Despliegue Continuo) que automatiza tareas cuando ocurren cambios en tu repositorio.

## Workflows Configurados

### 1. Tests (`test.yml`)
**Se ejecuta cuando:**
- Haces push a archivos en `tests/`
- Ejecutas manualmente desde GitHub

**Lo que hace:**
1. Verifica que todos los archivos necesarios existen
2. Ejecuta tests básicos sin necesidad de API keys
3. Valida la estructura del proyecto

### 2. Evaluación RAG (`eval.yml`)
**Se ejecuta cuando:**
- Haces push a archivos en `app/` o `tests/`
- Ejecutas manualmente desde GitHub

**Lo que hace:**
1. Verifica setup del proyecto
2. Si tienes `OPENAI_API_KEY` configurada: ejecuta evaluación completa
3. Si no tienes API key: salta la evaluación pero no falla

## Configuración de Secretos

### Para ejecutar evaluaciones completas necesitas configurar:

1. Ve a tu repositorio en GitHub
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Nombre: `OPENAI_API_KEY`
5. Valor: tu clave de OpenAI API

## Cómo ver los resultados

1. Ve a tu repositorio en GitHub
2. Pestaña "Actions"
3. Verás la lista de ejecuciones
4. Click en cualquier ejecución para ver detalles

## Estados posibles

- ✅ **Success**: Todo funcionó correctamente
- ❌ **Failure**: Hubo un error (revisa los logs)
- 🟡 **Skipped**: Se saltó la evaluación (falta API key)

## Solución de problemas comunes

### Error: "No module named 'app'"
- Verifica que el archivo `app/__init__.py` existe
- Verifica la estructura de directorios

### Error: "OPENAI_API_KEY not found"
- Configura el secreto en GitHub (ver arriba)
- O ignora este error si solo quieres ejecutar tests básicos

### Error: "Dataset not found"
- Verifica que `tests/eval_dataset.json` existe
- Verifica el formato del dataset

### Error: "Vectorstore not found"
- Verifica que el directorio `vectorstore/` existe
- Verifica que contiene `index.faiss` e `index.pkl`

## Comandos locales útiles

```bash
# Verificar setup antes de hacer push
python scripts/verify_setup.py

# Ejecutar tests localmente
pytest tests/test_run_eval.py -v

# Ejecutar evaluación localmente (necesita API key)
python app/run_eval.py
```

## Estructura de archivos necesaria

```
proyecto/
├── .github/workflows/
│   ├── test.yml
│   └── eval.yml
├── app/
│   ├── __init__.py
│   ├── rag_pipeline.py
│   ├── run_eval.py
│   └── prompts/
├── tests/
│   ├── test_run_eval.py
│   └── eval_dataset.json
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
├── scripts/
│   └── verify_setup.py
└── requirements.txt
```