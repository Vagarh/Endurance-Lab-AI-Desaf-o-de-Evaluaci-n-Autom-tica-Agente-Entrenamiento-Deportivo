# app/dashboard.py
import mlflow
import pandas as pd
import streamlit as st
import altair as alt   # ➟ Gráficos interactivos

########################################################################
# CONFIGURACIÓN GENERAL
########################################################################
st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# Conecta al tracking server por defecto (o define MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

########################################################################
# SELECCIÓN DE EXPERIMENTO
########################################################################
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]
if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar 📂", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=10_000,      # Ajusta si tu experimento tiene muchos runs
)

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

########################################################################
# PARSEAR DATOS DE LOS RUNS
########################################################################
CRITERIA = ["correctness", "relevance", "coherence", "toxicity", "harmfulness"]
# Si tus métricas llevan sufijo _score ajusta aquí ↓
METRIC_SUFFIX = "_score"

data = []
for run in runs:
    p = run.data.params
    m = run.data.metrics
    tags = run.data.tags

    row = {
        "pregunta"      : p.get("question"),
        "prompt_version": p.get("prompt_version"),
        "chunk_size"    : int(p.get("chunk_size", 0)),
        "chunk_overlap" : int(p.get("chunk_overlap", 0)),
        "run_id"        : run.info.run_id,
    }

    # Añadimos puntuaciones por criterio
    for c in CRITERIA:
        row[c] = m.get(f"{c}{METRIC_SUFFIX}", None)

    # (Opcional) razonamiento almacenado como tag o param
    row["razonamiento"] = tags.get("eval_reasoning") or p.get("eval_reasoning")

    data.append(row)

df = pd.DataFrame(data)

########################################################################
# TABLA COMPLETA CON FILTROS
########################################################################
st.subheader("📋 Resultados individuales por pregunta")
st.dataframe(df, use_container_width=True)

########################################################################
# AGRUPAR Y ANALIZAR
########################################################################
# Selección de criterio(s) para analizar
st.subheader("🎯 Selecciona criterio(s) para comparar")
selected_criteria = st.multiselect("Criterios:", CRITERIA, default=["correctness"])

if selected_criteria:
    # Agrupar por configuración
    grouped = (
        df.groupby(["prompt_version", "chunk_size"])
          [selected_criteria]
          .mean()
          .reset_index()
    )

    # Vista tabular
    st.subheader("📊 Desempeño agrupado por configuración")
    st.dataframe(grouped, use_container_width=True)

    # Configuración de etiquetas para el gráfico
    grouped["config"] = (
        grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)
    )

    # Gráfico interactivo con Altair
    chart_data = grouped.melt(
        id_vars=["config"], 
        value_vars=selected_criteria, 
        var_name="criterio", 
        value_name="score"
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("config:N", sort="-y", title="Configuración (prompt | chunk)"),
            y=alt.Y("score:Q", title="Puntuación media"),
            color="criterio:N",
            tooltip=["criterio", "score", "config"]
        )
        .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)

########################################################################
# MOSTRAR RAZONAMIENTOS (OPCIONAL)
########################################################################
if st.checkbox("Mostrar razonamientos del modelo 🧠"):
    st.subheader("💬 Razonamientos de la evaluación")
    # Seleccionar run para ver detalles
    sel_run_id = st.selectbox("Selecciona un run para ver su razonamiento:", df["run_id"])
    razon = df.loc[df["run_id"] == sel_run_id, "razonamiento"].values
    if razon.size and razon[0]:
        st.info(razon[0])
    else:
        st.warning("⚠️ El razonamiento no está disponible para este run.")

########################################################################
# NOTAS
########################################################################
st.markdown(
    """
    **Notas**  
    - Todas las métricas se calculan con la clase `LabeledCriteriaEvalChain`.  
    - Asegúrate de **registrar** cada métrica como `<criterio>_score` en MLflow para que se cargue correctamente.  
    - *Razonamientos* se esperan en el tag/param `eval_reasoning`; ajusta el nombre según tu pipeline.  
    - Si quieres comparar estadísticas adicionales (p. ej. desviación estándar) puedes extender la sección de agrupación.
    """
)
