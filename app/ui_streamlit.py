# app/ui_streamlit.py
import sys, os, traceback
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_ROOT))           # Para importar app.*

st.set_page_config(
    page_title="Asistente Deportivo · Endurance Coach",
    layout="wide",
    page_icon="🏃‍♂️",
)

from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# ------------------------------------------------------------------
# Estilos globales (burbujas de chat)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Funciones auxiliares con caché
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando base de conocimientos…")
def get_vectordb_and_chain():
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb)
    return vectordb, chain


# ------------------------------------------------------------------
# Barra lateral (configuración de la sesión)
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración de sesión")
sport = st.sidebar.selectbox(
    "Disciplina principal",
    ("Ciclismo", "Running", "Triatlón", "Natación", "Otro"),
    index=0,
)
st.sidebar.markdown(
    """
    <small>
    Elige tu disciplina para adaptar las recomendaciones
    de planificación, rendimiento y nutrición.
    </small>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ------------------------------------------------------------------
# Cabecera
# ------------------------------------------------------------------
st.title("🏆 Asistente de Entrenamiento Deportivo")
st.caption("Planificación · Análisis de rendimiento · Nutrición")

# Campo de entrada
question = st.text_input(
    "Formula tu pregunta (p. ej. «¿Cómo estructuro mi semana de carga antes de la carrera?»):"
)

# Historial en el estado de la sesión
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[tuple[str, str]] = []

# Cargar vectorstore y cadena solo una vez
_, chain = get_vectordb_and_chain()

# ------------------------------------------------------------------
# Llamada al modelo
# ------------------------------------------------------------------
if question:
    with st.spinner("Generando respuesta…"):
        try:
            result = chain.invoke(
                {
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                    "sport": sport,  # contexto adicional
                }
            )
            answer = result["answer"]
        except Exception:
            answer = (
                "⚠️ Lo siento, ha ocurrido un error al procesar tu pregunta.\n\n"
                f"```{traceback.format_exc(limit=2)}```"
            )
        # Guardar el turno
        st.session_state.chat_history.append((question, answer))

# ------------------------------------------------------------------
# Mostrar conversación
# ------------------------------------------------------------------
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
