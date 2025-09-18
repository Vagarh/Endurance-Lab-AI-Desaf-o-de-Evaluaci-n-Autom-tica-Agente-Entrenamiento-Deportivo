# app/main_interface.py
"""
Endurance Lab AI · Streamlit front‑end
Author  : Juan Felipe Cardona Arango  (github.com/Vagarh)
Repo    : https://github.com/Vagarh/Endurance-Lab-AI-Desaf-o-de-Evaluaci-n-Autom-tica-Agente-Entrenamiento-Deportivo
License : MIT
"""

# ───────────────────────────
#  Imports
# ───────────────────────────
import sys, os, traceback
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import mlflow

# ───────────────────────────
#  Rutas internas
# ───────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_ROOT))  # para importar app.*

from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# ───────────────────────────
#  Configuración del proyecto
# ───────────────────────────
try:
    from config import (
        PROJECT_NAME, PROJECT_VERSION, REPO_URL, 
        LOGO_PATH, HERO_PATH, SUPPORTED_SPORTS
    )
    LOGO_URL = str(LOGO_PATH) if LOGO_PATH.exists() else None
    HERO_URL = str(HERO_PATH) if HERO_PATH.exists() else None
    APP_VERSION = PROJECT_VERSION
except ImportError:
    # Fallback si no existe config.py
    REPO_URL = "https://github.com/Vagarh/Endurance-Lab-AI-Desaf-o-de-Evaluaci-n-Autom-tica-Agente-Entrenamiento-Deportivo"
    LOGO_URL = None
    HERO_URL = None
    APP_VERSION = "1.0.0"
    SUPPORTED_SPORTS = ["Ciclismo", "Running", "Triatlón", "Natación", "Otro"]

# ───────────────────────────
#  Configuración general
# ───────────────────────────
st.set_page_config(
    page_title="Endurance Lab AI · Chatbot & Métricas",
    layout="wide",
    page_icon="🏆",
    menu_items={
        "Get Help": REPO_URL,
        "Report a bug": REPO_URL + "/issues",
        "About": "Creado por Juan Felipe Cardona Arango · © {}".format(datetime.now().year),
    },
)

# ───────────────────────────
#  CSS global (burbujas + tipografía)
# ───────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
      html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
      .bubble-user{
        background:#e8f4fd;
        padding:.65em;
        border-radius:.6em;
        margin-bottom:.5em;
      }
      .bubble-bot{
        background:#f6f8fa;
        padding:.65em;
        border-radius:.6em;
        margin-bottom:1em;
      }
      footer {visibility:hidden;} /* ocultar footer default */
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────
#  Encabezado principal con hero
# ───────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(LOGO_URL, width=110)
with col_title:
    st.subheader("Endurance Lab AI")
    st.caption("Tu asistente virtual para entrenamiento de resistencia · v{}".format(APP_VERSION))

st.image(HERO_URL, width=200)

st.markdown("---")

# ───────────────────────────
#  Cache de recursos RAG
# ───────────────────────────
@st.cache_resource(show_spinner="🔄 Cargando base de conocimientos…")
def get_vectordb_and_chain():
    vectordb = load_vectorstore_from_disk()
    chain    = build_chain(vectordb)
    return vectordb, chain

# ───────────────────────────
#  Sidebar
# ───────────────────────────
with st.sidebar:
    st.image(LOGO_URL, width=90)
    st.markdown("### Navegación")
    modo = st.radio("", ["🤖 Chatbot", "📊 Métricas"], index=0)

    if modo == "🤖 Chatbot":
        sport = st.selectbox(
            "Disciplina principal",
            ("Ciclismo", "Running", "Triatlón", "Natación", "Otro"),
            index=0,
        )
        st.markdown("<small>Elige tu disciplina para adaptar las recomendaciones.</small>", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("#### Recursos")
    st.markdown(f"[Repositorio en GitHub]({REPO_URL})")
    st.markdown("[Reportar incidencia]({}/issues)".format(REPO_URL))
    st.markdown("---")
    st.markdown("<small>© {} Juan Felipe Cardona Arango</small>".format(datetime.now().year), unsafe_allow_html=True)

# ───────────────────────────
#  Cargar RAG una sola vez
# ───────────────────────────
_, chain = get_vectordb_and_chain()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                   CHAT                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if modo == "🤖 Chatbot":
    st.header("🗣️ Chat con tu Coach de Resistencia")

    pregunta = st.text_input("Formula tu pregunta (p. ej. ‘¿Cómo estructuro mi semana de carga antes de la carrera?’):")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[tuple[str, str]] = []

    if pregunta:
        with st.spinner("Generando respuesta…"):
            try:
                result  = chain.invoke({"question": pregunta, "chat_history": st.session_state.chat_history, "sport": sport})
                answer  = result["answer"]
            except Exception:
                answer  = "⚠️ Lo siento, ocurrió un error:\n\n```{}```".format(traceback.format_exc(limit=2))
            st.session_state.chat_history.append((pregunta, answer))

    if st.session_state.chat_history:
        st.markdown("---")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="bubble-user"><strong>🧑‍💻 Tú:</strong><br/>{q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-bot"><strong>🤖 Coach:</strong><br/>{a}</div>', unsafe_allow_html=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                  MÉTRICAS                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
else:  # modo == "📊 Métricas"
    st.header("📈 Resultados de Evaluación del LLM")

    try:
        client      = mlflow.tracking.MlflowClient()
        experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]
    except Exception as e:
        st.error(f"No se pudo conectar a MLflow: {e}")
        st.stop()

    if not experiments:
        st.warning("No se encontraron experimentos de evaluación.")
        st.stop()

    exp_names     = [exp.name for exp in experiments]
    selected_exp  = st.selectbox("Selecciona un experimento:", exp_names)

    experiment    = next(exp for exp in experiments if exp.name == selected_exp)
    runs          = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("No hay ejecuciones registradas.")
        st.stop()

    # ── DataFrame con parámetros y métricas
    data = [
        {
            "Pregunta"  : run.data.params.get("question"),
            "Prompt"    : run.data.params.get("prompt_version"),
            "Chunk Size": int(run.data.params.get("chunk_size", 0)),
            "Precisión" : run.data.metrics.get("lc_is_correct", np.nan),
        }
        for run in runs
    ]
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # ── Agrupar y limpiar
    grouped = (
        df.groupby(["Prompt", "Chunk Size"], dropna=False)
          .agg({"Precisión": "mean"})
          .reset_index()
    )
    grouped["Precisión"] = pd.to_numeric(grouped["Precisión"], errors="coerce")
    grouped = grouped.dropna(subset=["Precisión"])
    grouped = grouped[grouped["Precisión"].apply(np.isfinite)]

    if grouped.empty:
        st.info("No hay datos suficientes para graficar la precisión.")
    else:
        grouped["config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)

        chart = (
            alt.Chart(grouped)
               .mark_bar()
               .encode(
                   x=alt.X("config:N", title="Configuración", sort=None),
                   y=alt.Y("Precisión:Q", title="Precisión media", scale=alt.Scale(domain=[0,1])),
                   tooltip=["Prompt", "Chunk Size", alt.Tooltip("Precisión:Q", format=".2f")],
               )
               .properties(width=640, height=420)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small>Gráficos generados con Altair • Datos almacenados y versionados con MLflow • Código fuente en <a href='{}'>GitHub</a></small>".format(REPO_URL),
        unsafe_allow_html=True,
    )
