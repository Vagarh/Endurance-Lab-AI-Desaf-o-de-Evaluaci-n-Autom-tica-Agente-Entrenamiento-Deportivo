# app/main_interface.py
import sys, os, traceback
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np            # ← NUEVO
import altair as alt          # ← NUEVO
import mlflow

# ───────────────────────────
#  Rutas internas
# ───────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_ROOT))           # para importar app.*

from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# ───────────────────────────
#  Configuración general
# ───────────────────────────
st.set_page_config(
    page_title="Chatbot + Métricas · Endurance Coach",
    layout="wide",
    page_icon="🏆",
)

# ───────────────────────────
#  CSS global (burbujas)
# ───────────────────────────
st.markdown(
    """
    <style>
      .bubble-user{
        background:#e8f4fd;
        color:#000;
        padding:0.6em;
        border-radius:.5em;
        margin-bottom:.5em;
      }
      .bubble-bot{
        background:#f6f8fa;
        color:#000;
        padding:0.6em;
        border-radius:.5em;
        margin-bottom:1em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────
#  Cache de recursos RAG
# ───────────────────────────
@st.cache_resource(show_spinner="🔄 Cargando base de conocimientos…")
def get_vectordb_and_chain():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb)
    return vectordb, chain


# ───────────────────────────
#  Sidebar
# ───────────────────────────
modo = st.sidebar.radio("📂 Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])

if modo == "🤖 Chatbot":
    sport = st.sidebar.selectbox(
        "Disciplina principal",
        ("Ciclismo", "Running", "Triatlón", "Natación", "Otro"),
        index=0,
    )
    st.sidebar.markdown(
        "<small>Elige tu disciplina para adaptar las recomendaciones.</small>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

# ───────────────────────────
#  Cargar RAG una sola vez
# ───────────────────────────
_, chain = get_vectordb_and_chain()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                   CHAT                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if modo == "🤖 Chatbot":
    st.title("🏆 Asistente de Entrenamiento Deportivo")
    st.caption("Planificación · Análisis de rendimiento · Nutrición")

    pregunta = st.text_input(
        "Formula tu pregunta (p. ej. «¿Cómo estructuro mi semana de carga antes de la carrera?»):"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[tuple[str, str]] = []

    if pregunta:
        with st.spinner("Generando respuesta…"):
            try:
                result = chain.invoke(
                    {
                        "question": pregunta,
                        "chat_history": st.session_state.chat_history,
                        "sport": sport,
                    }
                )
                answer = result["answer"]
            except Exception:
                answer = (
                    "⚠️ Lo siento, ocurrió un error al procesar tu pregunta.\n\n"
                    f"```{traceback.format_exc(limit=2)}```"
                )
            st.session_state.chat_history.append((pregunta, answer))

    if st.session_state.chat_history:
        st.markdown("---")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(
                f'<div class="bubble-user"><strong>🧑‍💻 Tú:</strong><br/>{q}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="bubble-bot"><strong>🤖 Coach:</strong><br/>{a}</div>',
                unsafe_allow_html=True,
            )

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                  MÉTRICAS                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
elif modo == "📊 Métricas":
    st.title("📈 Resultados de Evaluación del LLM")

    try:
        client = mlflow.tracking.MlflowClient()
        experiments = [
            exp for exp in client.search_experiments()
            if exp.name.startswith("eval_")
        ]
    except Exception as e:
        st.error(f"No se pudo conectar a MLflow: {e}")
        st.stop()

    if not experiments:
        st.warning("No se encontraron experimentos de evaluación.")
        st.stop()

    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("Selecciona un experimento:", exp_names)

    experiment = next(exp for exp in experiments if exp.name == selected_exp)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if not runs:
        st.warning("No hay ejecuciones registradas.")
        st.stop()

    # ── DataFrame con parámetros y métricas
    data = []
    for run in runs:
        params  = run.data.params
        metrics = run.data.metrics
        data.append(
            {
                "Pregunta":     params.get("question"),
                "Prompt":       params.get("prompt_version"),
                "Chunk Size":   int(params.get("chunk_size", 0)),
                "Precisión":    metrics.get("lc_is_correct", np.nan),
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # ── Agrupar y limpiar
    grouped = (
        df.groupby(["Prompt", "Chunk Size"])
        .agg({"Precisión": "mean"})
        .reset_index()
    )

    # ── Conversión numérica y limpieza de NaN/Inf
    grouped["Precisión"] = pd.to_numeric(grouped["Precisión"], errors="coerce")
    grouped = grouped.dropna(subset=["Precisión"])
    grouped = grouped[grouped["Precisión"].apply(np.isfinite)]

    if grouped.empty:
        st.info("No hay datos suficientes para graficar la precisión.")
    else:
        grouped["config"] = (
            grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
        )

        # ── Gráfico robusto con Altair
        chart = (
            alt.Chart(grouped)
            .mark_bar()
            .encode(
                x=alt.X("config:N", title="Configuración", sort=None),
                y=alt.Y("Precisión:Q", title="Precisión media", scale=alt.Scale(domain=[0,1])),
                tooltip=["Prompt", "Chunk Size", alt.Tooltip("Precisión:Q", format=".2f")],
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, use_container_width=True)
