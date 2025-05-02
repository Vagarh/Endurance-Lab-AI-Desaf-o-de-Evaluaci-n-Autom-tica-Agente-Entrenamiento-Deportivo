# app/main_interface.py
import sys, os, traceback
from pathlib import Path
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Preparación de rutas e importaciones internas
# ──────────────────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_ROOT))           # Para importar app.*

from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402
import pandas as pd
import mlflow

# ──────────────────────────────────────────────────────────────────────────────
# Configuración general de la página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chatbot + Métricas · Endurance Coach",
    layout="wide",
    page_icon="🏆",
)

# ──────────────────────────────────────────────────────────────────────────────
# Estilos globales (burbujas de chat)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Recursos con caché (vectorstore + chain)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Cargando base de conocimientos…")
def get_vectordb_and_chain():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb)
    return vectordb, chain


# ──────────────────────────────────────────────────────────────────────────────
# Barra lateral – selección de modo y parámetros
# ──────────────────────────────────────────────────────────────────────────────
modo = st.sidebar.radio("📂 Selecciona una vista:", ["🤖 Chatbot", "📊 Métricas"])

if modo == "🤖 Chatbot":
    # Parámetro adicional para personalizar las respuestas
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

# ──────────────────────────────────────────────────────────────────────────────
# Obtención (única) de recursos RAG
# ──────────────────────────────────────────────────────────────────────────────
_, chain = get_vectordb_and_chain()

# ──────────────────────────────────────────────────────────────────────────────
# Modo 1: Chatbot
# ──────────────────────────────────────────────────────────────────────────────
if modo == "🤖 Chatbot":
    st.title("🏆 Asistente de Entrenamiento Deportivo")
    st.caption("Planificación · Análisis de rendimiento · Nutrición")

    pregunta = st.text_input(
        "Formula tu pregunta (p. ej. «¿Cómo estructuro mi semana de carga antes de la carrera?»):"
    )

    # Estado de conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[tuple[str, str]] = []

    # Llamada al modelo
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

    # Render del historial
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

# ──────────────────────────────────────────────────────────────────────────────
# Modo 2: Métricas con MLflow
# ──────────────────────────────────────────────────────────────────────────────
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

    # Construir DataFrame con parámetros y métricas clave
    data = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        data.append(
            {
                "Pregunta": params.get("question"),
                "Prompt": params.get("prompt_version"),
                "Chunk Size": int(params.get("chunk_size", 0)),
                "Correcto (LC)": metrics.get("lc_is_correct", 0),
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Agrupado por configuración
    st.subheader("📊 Precisión promedio por configuración")
    grouped = (
        df.groupby(["Prompt", "Chunk Size"])
        .agg({"Correcto (LC)": "mean"})
        .reset_index()
    )
    grouped.rename(columns={"Correcto (LC)": "Precisión"}, inplace=True)
    grouped["config"] = (
        grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
    )
    st.bar_chart(grouped.set_index("config")["Precisión"])

