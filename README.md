````markdown
<p align="center">
  <img src="assets/banner.png" alt="Endurance Lab AI banner" style="max-width:100%;height:auto;border-radius:6px;">
</p>

<p align="center">
  <img src="assets/logo.png" alt="Endurance Lab AI logo" width="160">
</p>

<h1 align="center">Endurance Lab AI 🏆</h1>
<p align="center">Asistente virtual para <strong>entrenamiento de resistencia</strong> — ciclismo · running · triatlón.</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Vagarh/Endurance-Lab-AI-Desaf-o-de-Evaluaci-n-Autom-tica-Agente-Entrenamiento-Deportivo?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Streamlit-1.34+-brightgreen?style=flat-square" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-0.2+-violet?style=flat-square" alt="LangChain">
  <img src="https://img.shields.io/badge/MLflow-2.12+-blue?style=flat-square" alt="MLflow">
</p>

---

> **Autor:** Juan Felipe Cardona Arango   ·   **Fecha:** mayo 2025  
> **Cursos:** *Procesamiento de Lenguaje Natural* y *Experiencias en Inteligencia de Negocios*  
> **Docentes:** Juan David Martínez Vargas · Ana María López Moreno · Edwin Nelson Montoya Múnera  
> **Ítem:** Proyecto final de ambos cursos

---

## 📖 Descripción

Este repositorio contiene la implementación y evaluación de **Endurance Lab AI**, un asistente virtual experto en entrenamiento de resistencia. Incluye:

- ⚙️ **Personalización** de dominio y prompts.  
- ✅ **Conjunto de pruebas** `eval_dataset.csv`.  
- 🤖 **Evaluación automática** con LangChain + MLflow.  
- 📊 **Dashboard interactivo** de métricas.  
- 📚 **Reflexiones** y análisis comparativo.  
- ✨ Criterio adicional de **Claridad**.

---

## 🗂️ Tabla de Contenidos

1. [Parte 1 · Personalización](#parte-1-personalización)  
2. [Parte 2 · Evaluación Automática](#parte-2-evaluación-automática)  
3. [Parte 3 · Reto Investigador](#parte-3-reto-investigador)  
4. [Parte 4 · Dashboard](#parte-4-dashboard)  
5. [Parte 5 · Presentación y Reflexión](#parte-5-presentación-y-reflexión)  
6. [🚀 Bonus · Criterio “Claridad”](#-bonus-criterio-claridad)  
7. [🛠️ Cómo Ejecutar](#️-cómo-ejecutar)  
8. [📄 Licencia](#-licencia)

---

## Parte 1 · Personalización

### 1. Dominio 🏅
* Entrenamiento deportivo de resistencia.  
* Especialidades: **ciclismo, triatlón, natación, running**.

### 2. Documentos Internos 📂
Se sustituyeron los PDFs base por:

| Tipo de documento          | Archivo                         |
|----------------------------|---------------------------------|
| Planes de entrenamiento    | `planes_entrenamiento.pdf`      |
| Historiales de rendimiento | `historiales_rendimiento.pdf`   |
| Guías de nutrición         | `guias_nutricion.pdf`           |
| Revisiones bibliográficas  | `revisiones_bibliograficas.pdf` |

### 3. Prompts 📝
<details>
<summary>Prompt principal</summary>

```text
SISTEMA:
Eres Endurance Lab AI, un asistente virtual experto en entrenamiento de resistencia…
RESPONSABILIDADES:
1. Validar contexto…
2. Solicitar datos faltantes…
3. Responder con cita…
FORMATO:
- USUARIO: {question}
- CONTEXTO: {context}
- ASISTENTE: …
````

</details>

<details>
<summary>Prompt secundario (breve y directo)</summary>

```text
SISTEMA:
Eres Endurance Lab AI. Responde solo con información interna.
- Sé breve…
```

</details>

### 4. Conjunto de pruebas 🧪

* Archivo: `eval_dataset.csv`
* Cobertura de casos típicos 🖉 y extremos 🛠️.

---

## Parte 2 · Evaluación Automática

| Recurso       | Descripción                                                                                |
| ------------- | ------------------------------------------------------------------------------------------ |
| `run_eval.py` | Script de evaluación ⚙️                                                                    |
| **LangChain** | `LabeledCriteriaEvalChain` para *correctness, relevance, coherence, toxicity, harmfulness* |
| **MLflow**    | Registro de métricas y artefactos 📈                                                       |

---

## Parte 3 · Reto Investigador

> Se añadieron criterios avanzados; cada uno guarda métrica `*_score` y razonamiento `*_reasoning`.

---

## Parte 4 · Dashboard

| Archivo                 | Funcionalidad                                                |
| ----------------------- | ------------------------------------------------------------ |
| `dashboard.py`          | Gráficos de métricas por criterio 📊                         |
| `app/main_interface.py` | Selector de experimentos, tabla interactiva, gráficas Altair |

---

## Parte 5 · Presentación y Reflexión

| Configuración          | Correctness | Relevance | Coherence | Toxicity | Harmfulness |
| ---------------------- | :---------: | :-------: | :-------: | :------: | :---------: |
| Chunk = 512 · Prompt A |   **0.87**  |    0.85   |    0.82   |   0.00   |     0.00    |
| Chunk = 256 · Prompt B |   **0.92**  |    0.90   |    0.88   |   0.00   |     0.00    |

* 🎯 **Mejor**: *chunk 256 + Prompt B*
* ⚠️ Fallos: contexto incompleto, citas omitidas, formateo inconsistente.
* 🧩 Toxicidad/Incoherencia: **0 %** toxicidad; leves incoherencias en chunks grandes.

---

## 🚀 Bonus · Criterio “Claridad”

Mide la facilidad de comprensión para deportistas (fluidez, estructura, jerga mínima).

```python
from langchain.evaluation.criteria.eval_chain import LabeledCriteriaEvalChain

criteria = [
    {"name": "clarity", "description": "¿La respuesta es clara y fácil de entender para un deportista?"},
]

eval_chain = LabeledCriteriaEvalChain.from_criteria(
    llm=llm,
    criteria=criteria,
    input_key="response",
    prediction_key="score",
)
```

Métrica en MLflow: **`clarity_score`** + artifact **`clarity_reasoning`**.

---

## 🛠️ Cómo Ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar evaluación automática
python run_eval.py --dataset eval_dataset.csv

# 3. Iniciar la app Streamlit
streamlit run app/main_interface.py
```

*La app estará disponible en [http://localhost:8501](http://localhost:8501).*

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Consulta el archivo [`LICENSE`](LICENSE) para más detalles.

```
```
