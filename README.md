Endurance Lab AI 🏆
Asistente virtual para entrenamiento de resistencia — ciclismo · running · triatlón

Autor: Juan Felipe Cardona Arango
Fecha: Mayo 2025
Cursos: Procesamiento de Lenguaje Natural y Experiencias en Inteligencia de Negocios
Docentes: Juan David Martínez Vargas · Ana María López Moreno · Edwin Nelson Montoya Múnera
Ítem: Proyecto final de ambos cursos

------------------------------------------------------------

📖 Descripción

Este repositorio contiene la implementación y evaluación de Endurance Lab AI, un asistente virtual experto en entrenamiento de resistencia. Incluye:

- Personalización del dominio y los prompts
- Conjunto de pruebas "eval_dataset.csv"
- Evaluación automática con LangChain y MLflow
- Dashboard interactivo de métricas
- Reflexiones y análisis comparativo
- Criterio adicional de Claridad

------------------------------------------------------------

🗂️ Tabla de Contenidos

1. Parte 1: Personalización
2. Parte 2: Evaluación Automática
3. Parte 3: Reto Investigador
4. Parte 4: Dashboard
5. Parte 5: Presentación y Reflexión
6. Bonus: Claridad
7. Cómo Ejecutar
8. Licencia

------------------------------------------------------------

Parte 1: Personalización

1. Dominio
Asistente especializado en entrenamiento de resistencia: ciclismo, triatlón, natación y running.

2. Documentos internos utilizados:
- planes_entrenamiento.pdf
- historiales_rendimiento.pdf
- guias_nutricion.pdf
- revisiones_bibliograficas.pdf

3. Prompts utilizados:

Prompt principal:
- SISTEMA: Eres Endurance Lab AI, un asistente virtual experto...
- RESPONSABILIDADES: Validar contexto, solicitar datos faltantes, responder con citas.
- FORMATO: USUARIO {question}, CONTEXTO {context}, ASISTENTE ...

Prompt secundario:
- SISTEMA: Responde solo con información interna. Sé breve y directo.

4. Conjunto de pruebas:
- Archivo: eval_dataset.csv
- Casos típicos y extremos

------------------------------------------------------------

Parte 2: Evaluación Automática

- Script: run_eval.py
- Evaluador: LangChain (correctness, relevance, coherence, toxicity, harmfulness)
- Registro: MLflow

------------------------------------------------------------

Parte 3: Reto Investigador

Se agregaron métricas avanzadas:
- *_score: valor numérico
- *_reasoning: justificación textual

------------------------------------------------------------

Parte 4: Dashboard

- dashboard.py: visualización por criterio
- app/main_interface.py: selector de experimentos y gráficos interactivos

------------------------------------------------------------

Parte 5: Presentación y Reflexión

Resultados por configuración:

| Chunk Size | Prompt  | Correctness | Relevance | Coherence | Toxicity | Harmfulness |
|------------|---------|-------------|-----------|-----------|----------|-------------|
| 512        | A       | 0.87        | 0.85      | 0.82      | 0.00     | 0.00        |
| 256        | B       | 0.92        | 0.90      | 0.88      | 0.00     | 0.00        |

Observaciones:
- Mejor configuración: Chunk 256 + Prompt B
- Fallos detectados: contexto incompleto, formato inconsistente, citas ausentes

------------------------------------------------------------

Bonus: Claridad

Evaluación de la claridad para usuarios deportistas (fluidez, jerga, estructura)

Código base (LangChain):
- clarity_score: puntuación
- clarity_reasoning: justificación

------------------------------------------------------------

🛠 Cómo Ejecutar

1. Instalar dependencias:
   pip install -r requirements.txt

2. Ejecutar evaluación:
   python run_eval.py --dataset eval_dataset.csv

3. Lanzar aplicación Streamlit:
   streamlit run app/main_interface.py

Accede desde tu navegador en http://localhost:8501

------------------------------------------------------------

📄 Licencia

Este proyecto está licenciado bajo los términos de la Licencia MIT.
Consulta el archivo LICENSE para más detalles.

