# Endurance Lab AI — Desafío de Evaluación Automática Agente Entrenamiento Deportivo.

Autor: Juan Felipe Cardona Arango  
Fecha: Mayo 2025
Cursos: Procesamiento de Lenguaje Natural y Experiencias en Inteligencia de Negocios
Docentes: Juan David Martinez Vargas, Ana Maria Lopez Moreno, Edwin Nelson Montoya Munera.
Item: Proyecto final de cursos.
---

📖 Descripción

Este repositorio contiene la implementación y evaluación de Endurance Lab AI, un asistente virtual experto en entrenamiento de resistencia (ciclismo, running, triatlón). Incluye:

- Personalización del dominio y prompts
- Conjunto de pruebas (eval_dataset.csv)
- Integración de evaluación automática con LangChain y MLflow
- Dashboard interactivo de métricas
- Reflexiones y análisis comparativo
- Criterio adicional de “Claridad”

---

📑 Tabla de Contenidos

1. Parte 1: Personalización  
2. Parte 2: Evaluación Automática  
3. Parte 3: Reto Investigador  
4. Parte 4: Dashboard  
5. Parte 5: Presentación y Reflexión  
6. 🚀 Bonus: Criterio “Claridad”  
7. 🛠️ Cómo ejecutar  
8. 📄 Licencia  

---

Parte 1: Personalización

1. Dominio
- Dominio de entrenamiento deportivo de resistencia  
- Especialidades: ciclismo, triatlón, natación, running  

2. Documentos Internos
Se reemplazaron los PDFs base por:

- planes_entrenamiento.pdf  
- historiales_rendimiento.pdf  
- guias_nutricion.pdf  
- revisiones_bibliograficas.pdf  

3. Prompts

Prompt Principal
SISTEMA:
Eres Endurance Lab AI, un asistente virtual experto en entrenamiento de resistencia (ciclismo, running, triatlón).
Utiliza solo estos documentos internos:
- Planes de entrenamiento
- Historiales de rendimiento
- Guías de nutrición
- Revisiones bibliográficas

RESPONSABILIDADES:
1. Validar contexto (plan actual, objetivos, nivel físico, restricciones, disponibilidad).
2. Solicitar datos faltantes con preguntas breves.
3. Responder con cita de fuente interna; no hacer conjeturas.
4. Tono técnico, empático y motivador.

FORMATO:
- USUARIO: “{question}”
- CONTEXTO: “{context}”
- ASISTENTE:
  1. Comprobación de contexto
  2. Petición de información (si aplica)
  3. Respuesta (con cita)

Prompt Secundario (Breve y Directo)
SISTEMA:
Eres Endurance Lab AI. Responde solo con información interna.
- Sé breve y directo.
- Si falta información, responde: “No tengo información suficiente”.

FORMATO:
- Pregunta: {question}
- Contexto: {context}
- Respuesta:

4. Conjunto de Pruebas
- Archivo: eval_dataset.csv
- Cobertura de casos típicos y extremos.

---

Parte 2: Evaluación Automática

- Script de evaluación: run_eval.py
- Integración con LangChain:
  - LabeledCriteriaEvalChain para criterios:
    - correctness
    - relevance
    - coherence
    - toxicity
    - harmfulness
- Reporte de métricas en MLflow.

---

Parte 3: Reto Investigador

- Añadidos criterios avanzados usando LabeledCriteriaEvalChain
- Cada criterio:
  - Métrica (*_score)
  - Razonamiento opcional como artifact (*_reasoning)

---

Parte 4: Dashboard

- Archivos modificados:
  - dashboard.py
  - main_interface.py
- Funcionalidades:
  - Gráficos de métricas por criterio
  - Selector para comparar criterios
  - Visualización de razonamientos opcionales

---

Parte 5: Presentación y Reflexión

Comparación de configuraciones:

Configuración             | Correctness | Relevance | Coherence | Toxicity | Harmfulness
---------------------------|-----------:|----------:|----------:|---------:|------------:
Chunk=512 + Prompt A      |        0.87 |      0.85 |      0.82 |     0.00 |        0.00
Chunk=256 + Prompt B      |        0.92 |      0.90 |      0.88 |     0.00 |        0.00

- Mejor configuración: chunk=256 + Prompt B
- Fallos detectados: contexto incompleto, citas omitidas, formateo inconsistente.
- Toxicidad/Incoherencia: cero toxicidad; algunas incoherencias en chunks grandes.

---

🚀 Bonus: Criterio “Claridad”

Descripción: mide la facilidad de comprensión para deportistas (fluidez, estructura, jerga mínima).

Código de implementación con LabeledCriteriaEvalChain:
```python
from langchain.evaluation.criteria.eval_chain import LabeledCriteriaEvalChain

criteria = [
    {"name": "clarity", "description": "¿La respuesta es clara y fácil de entender para un deportista?"},
    # demás criterios...
]

eval_chain = LabeledCriteriaEvalChain.from_criteria(
    llm=llm,
    criteria=criteria,
    input_key="response",
    prediction_key="score"
)
```

- Métrica en MLflow: clarity_score
- Artifact: clarity_reasoning

---

🛠️ Cómo ejecutar

1. Instalar dependencias
   pip install -r requirements.txt
2. Ejecutar evaluación
   python run_eval.py --dataset eval_dataset.csv
3. Iniciar dashboard
   python dashboard.py

---

📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta LICENSE para más detalles.
