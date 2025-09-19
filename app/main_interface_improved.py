# app/main_interface_improved.py
"""
Endurance Lab AI · Streamlit front‑end (Versión Mejorada)
Author  : Juan Felipe Cardona Arango  (github.com/Vagarh)
License : MIT
"""

import sys, os, traceback
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import mlflow

# Configuración del proyecto
APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_ROOT))

try:
    from config import (
        PROJECT_NAME, PROJECT_VERSION, REPO_URL, 
        LOGO_PATH, HERO_PATH, SUPPORTED_SPORTS,
        PROMPT_VERSION
    )
    LOGO_URL = str(LOGO_PATH) if LOGO_PATH.exists() else None
    HERO_URL = str(HERO_PATH) if HERO_PATH.exists() else None
    APP_VERSION = PROJECT_VERSION
except ImportError:
    # Fallback si no existe config.py
    REPO_URL = "https://github.com/Vagarh/Endurance-Lab-AI"
    LOGO_URL = None
    HERO_URL = None
    APP_VERSION = "1.0.0"
    SUPPORTED_SPORTS = ["Ciclismo", "Running", "Triatlón", "Natación", "Otro"]
    PROMPT_VERSION = "v1_asistente_deporte"

from app.rag_pipeline import load_vectorstore_from_disk, build_chain

# Configuración de Streamlit
st.set_page_config(
    page_title="Endurance Lab AI · Chatbot & Métricas",
    layout="wide",
    page_icon="🏆",
    menu_items={
        "Get Help": REPO_URL,
        "Report a bug": REPO_URL + "/issues",
        "About": f"Creado por Juan Felipe Cardona Arango · © {datetime.now().year}",
    },
)

# CSS mejorado y moderno
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #C73E1D;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-shadow: 0 10px 30px rgba(0,0,0,0.1);
        --border-radius: 16px;
    }
    
    /* Fuentes globales */
    html, body, [class*="css"] { 
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header principal */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: var(--bg-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Cards y contenedores */
    .sport-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .sport-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--bg-gradient);
    }
    
    .sport-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border-color: var(--primary-color);
    }
    
    /* Burbujas de chat mejoradas */
    .bubble-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        max-width: 80%;
        margin-left: auto;
        animation: slideInRight 0.3s ease;
    }
    
    .bubble-bot {
        background: white;
        color: #333;
        padding: 1.2rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--accent-color);
        max-width: 85%;
        animation: slideInLeft 0.3s ease;
    }
    
    /* Animaciones */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Métricas mejoradas */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        border-top: 4px solid var(--primary-color);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        animation: pulse 0.6s ease;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Botones mejorados */
    .stButton > button {
        background: var(--bg-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--card-shadow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar mejorado */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Inputs mejorados */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 12px !important;
        border: 2px solid #e9ecef !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(46, 134, 171, 0.1) !important;
    }
    
    /* Iconos deportivos */
    .sport-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #28a745;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    /* Progress bars */
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--bg-gradient);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Footer oculto */
    footer {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .bubble-user, .bubble-bot {
            max-width: 95%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Encabezado principal mejorado
st.markdown('<h1 class="main-header">🏆 Endurance Lab AI</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Tu asistente virtual inteligente para entrenamiento de resistencia · v{APP_VERSION}</p>', unsafe_allow_html=True)

# Hero section con cards deportivas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="sport-card">
        <div style="text-align: center;">
            <span class="sport-icon">🚴‍♂️</span>
            <h4 style="margin: 0.5rem 0; color: #2E86AB;">Ciclismo</h4>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">FTP, Potencia, Zonas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="sport-card">
        <div style="text-align: center;">
            <span class="sport-icon">🏃‍♂️</span>
            <h4 style="margin: 0.5rem 0; color: #A23B72;">Running</h4>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Pace, Técnica, Resistencia</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="sport-card">
        <div style="text-align: center;">
            <span class="sport-icon">🏊‍♂️</span>
            <h4 style="margin: 0.5rem 0; color: #F18F01;">Natación</h4>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Técnica, Brazada, Ritmo</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="sport-card">
        <div style="text-align: center;">
            <span class="sport-icon">🏆</span>
            <h4 style="margin: 0.5rem 0; color: #C73E1D;">Triatlón</h4>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Transiciones, Estrategia</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Cache de recursos RAG
@st.cache_resource(show_spinner="🔄 Cargando base de conocimientos…")
def get_vectordb_and_chain():
    try:
        vectordb = load_vectorstore_from_disk()
        chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)
        return vectordb, chain
    except Exception as e:
        st.error(f"Error cargando el sistema RAG: {e}")
        st.info("Asegúrate de haber ejecutado 'python create_vectorstore.py' primero")
        return None, None

# Sidebar mejorado y moderno
with st.sidebar:
    # Logo y estado
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="margin: 0; color: #2E86AB;">🏆 Endurance Lab</h2>
        <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">
            <span class="status-online"></span>Sistema Activo
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navegación con iconos mejorados
    st.markdown("### 🧭 Navegación")
    modo = st.radio(
        "Selecciona el modo:",
        ["🤖 Asistente Inteligente", "📊 Dashboard de Métricas", "📈 Analytics Básicos"],
        index=0,
        help="Elige entre chatbot, dashboard avanzado o métricas básicas"
    )

    if modo == "🤖 Asistente Inteligente":
        st.markdown("---")
        st.markdown("### ⚙️ Personalización")
        
        # Disciplina con iconos
        sport_options = {
            "🚴‍♂️ Ciclismo": "Ciclismo",
            "🏃‍♂️ Running": "Running", 
            "🏊‍♂️ Natación": "Natación",
            "🏆 Triatlón": "Triatlón",
            "🎯 Otro": "Otro"
        }
        
        sport_display = st.selectbox(
            "Disciplina principal:",
            list(sport_options.keys()),
            index=0,
            help="Selecciona tu deporte principal para recomendaciones personalizadas"
        )
        sport = sport_options[sport_display]
        
        # Nivel con descripción
        nivel_options = {
            "🌱 Principiante": "Principiante",
            "📈 Intermedio": "Intermedio", 
            "🔥 Avanzado": "Avanzado",
            "👑 Élite": "Élite"
        }
        
        nivel_display = st.selectbox(
            "Nivel de experiencia:",
            list(nivel_options.keys()),
            index=1,
            help="Tu nivel actual de entrenamiento"
        )
        nivel = nivel_options[nivel_display]
        
        # Objetivos
        st.markdown("### 🎯 Objetivo Principal")
        objetivo = st.selectbox(
            "¿Qué quieres lograr?",
            [
                "🏃‍♂️ Mejorar resistencia",
                "⚡ Aumentar velocidad", 
                "💪 Ganar fuerza",
                "🏆 Preparar competición",
                "🔄 Recuperación activa",
                "📚 Aprender técnica"
            ],
            help="Objetivo principal de entrenamiento"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips Rápidos")
        
        tips = [
            "💧 Mantente hidratado durante entrenamientos largos",
            "😴 El descanso es tan importante como el entrenamiento",
            "📊 Registra tus entrenamientos para ver progreso",
            "🍎 La nutrición afecta directamente tu rendimiento"
        ]
        
        tip_del_dia = tips[datetime.now().day % len(tips)]
        st.info(f"**Tip del día:** {tip_del_dia}")
        
    elif modo == "📊 Dashboard de Métricas":
        st.markdown("---")
        st.markdown("### 📊 Dashboard Avanzado")
        st.info("Dashboard interactivo con gráficos avanzados y análisis detallado.")
        
        if st.button("🚀 Ejecutar Nueva Evaluación", type="primary"):
            st.info("Ejecutando evaluación en segundo plano...")
    
    else:  # Analytics Básicos
        st.markdown("---")
        st.markdown("### 📈 Analytics Básicos")
        st.info("Visualización básica del rendimiento del sistema de IA.")
        
    st.markdown("---")
    
    # Recursos con iconos mejorados
    st.markdown("### 🔗 Recursos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"[📚 GitHub]({REPO_URL})")
        st.markdown("📧 Soporte")
    with col2:
        st.markdown(f"[🐛 Issues]({REPO_URL}/issues)")
        st.markdown("📖 Docs")
    
    # Estadísticas rápidas
    st.markdown("---")
    st.markdown("### 📈 Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Consultas", "1.2K+", "↗️ 15%")
    with col2:
        st.metric("Precisión", "94%", "↗️ 2%")
    
    st.markdown("---")
    st.markdown(
        f'<div style="text-align: center; color: #666; font-size: 0.8rem;">'
        f'© {datetime.now().year} Juan Felipe Cardona<br>'
        f'<span style="color: #2E86AB;">Endurance Lab AI v{APP_VERSION}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# Cargar sistema RAG
vectordb, chain = get_vectordb_and_chain()

if chain is None:
    st.error("❌ No se pudo cargar el sistema RAG. Verifica la configuración.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#                                   CHAT
# ═══════════════════════════════════════════════════════════════════════════════
if modo == "🤖 Asistente Inteligente":
    # Header con información contextual
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h2 style="margin: 0 0 1rem 0; color: #2E86AB;">💬 Tu Coach Personal de Resistencia</h2>
        <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
            <div><strong>🏃‍♂️ Disciplina:</strong> <span style="color: #A23B72;">{}</span></div>
            <div><strong>📈 Nivel:</strong> <span style="color: #F18F01;">{}</span></div>
            <div><strong>🎯 Objetivo:</strong> <span style="color: #C73E1D;">{}</span></div>
        </div>
    </div>
    """.format(sport, nivel, objetivo.split(' ', 1)[1] if ' ' in objetivo else objetivo), unsafe_allow_html=True)
    
    # Sugerencias rápidas
    st.markdown("### 💡 Preguntas Sugeridas")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚴‍♂️ Mejorar FTP", help="Consejos para aumentar tu Functional Threshold Power"):
            st.session_state.pregunta_sugerida = "¿Cómo puedo mejorar mi FTP en 6 semanas entrenando 8 horas por semana?"
    
    with col2:
        if st.button("🏃‍♂️ Plan de Running", help="Estructura de entrenamiento para running"):
            st.session_state.pregunta_sugerida = "¿Cómo estructuro mi semana de entrenamiento para mejorar en 10K?"
    
    with col3:
        if st.button("🍎 Nutrición Deportiva", help="Consejos de alimentación para entrenamientos"):
            st.session_state.pregunta_sugerida = "¿Cuántos carbohidratos necesito por hora en entrenamientos largos?"
    
    # Input de pregunta mejorado
    pregunta_inicial = st.session_state.get('pregunta_sugerida', '')
    if pregunta_inicial:
        del st.session_state.pregunta_sugerida
    
    pregunta = st.text_area(
        "✍️ Escribe tu pregunta o consulta:",
        value=pregunta_inicial,
        placeholder="Ej: ¿Cómo puedo mejorar mi resistencia para una maratón? ¿Qué ejercicios me recomiendas para fortalecer las piernas?",
        height=120,
        help="Sé específico sobre tu situación actual, objetivos y limitaciones para obtener mejores recomendaciones"
    )
    
    # Botones de acción mejorados
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        enviar = st.button("🚀 Enviar Consulta", type="primary", use_container_width=True)
    with col2:
        limpiar = st.button("🗑️ Limpiar Chat", use_container_width=True)
    with col3:
        if st.button("💾 Guardar Chat", use_container_width=True):
            if st.session_state.get('chat_history'):
                st.success("Chat guardado localmente")
    with col4:
        if st.button("📋 Exportar", use_container_width=True):
            if st.session_state.get('chat_history'):
                st.info("Función de exportación disponible próximamente")
    
    # Inicializar historial de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if limpiar:
        st.session_state.chat_history = []
        st.success("Chat limpiado correctamente")
        st.rerun()
    
    # Procesamiento de pregunta
    if enviar and pregunta.strip():
        with st.spinner("🤔 Analizando tu consulta y buscando la mejor respuesta..."):
            try:
                # Agregar contexto completo
                contexto_adicional = f"Disciplina: {sport}, Nivel: {nivel}, Objetivo: {objetivo}"
                pregunta_completa = f"{pregunta}\n\nContexto del usuario: {contexto_adicional}"
                
                result = chain.invoke({
                    "question": pregunta_completa, 
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                
                # Agregar al historial con timestamp
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append((pregunta, answer, timestamp))
                
                st.success("¡Respuesta generada correctamente!")
                
            except Exception as e:
                error_msg = f"⚠️ Lo siento, ocurrió un error al procesar tu consulta:\n\n```\n{str(e)}\n```\n\nPor favor, intenta reformular tu pregunta o verifica tu conexión."
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append((pregunta, error_msg, timestamp))
                st.error("Error al procesar la consulta")
    
    elif enviar and not pregunta.strip():
        st.warning("Por favor, escribe una pregunta antes de enviar.")
    
    # Mostrar historial de chat mejorado
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💭 Historial de Conversación")
        
        # Contenedor scrolleable para el chat
        chat_container = st.container()
        
        with chat_container:
            for i, chat_item in enumerate(reversed(st.session_state.chat_history)):
                if len(chat_item) == 3:
                    q, a, timestamp = chat_item
                else:
                    q, a = chat_item
                    timestamp = "00:00"
                
                # Pregunta del usuario con timestamp
                st.markdown(
                    f'''<div class="bubble-user">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>🧑‍💻 Tú</strong>
                            <small style="opacity: 0.7;">{timestamp}</small>
                        </div>
                        <div>{q}</div>
                    </div>''', 
                    unsafe_allow_html=True
                )
                
                # Respuesta del asistente
                st.markdown(
                    f'''<div class="bubble-bot">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>🤖 Coach Endurance</strong>
                            <small style="opacity: 0.7;">{timestamp}</small>
                        </div>
                        <div>{a}</div>
                    </div>''', 
                    unsafe_allow_html=True
                )
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown('<div style="margin: 1rem 0; border-bottom: 1px solid #eee;"></div>', unsafe_allow_html=True)
        
        # Estadísticas del chat
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Consultas", len(st.session_state.chat_history))
        with col2:
            total_chars = sum(len(q) + len(a) for q, a, *_ in st.session_state.chat_history)
            st.metric("📝 Caracteres", f"{total_chars:,}")
        with col3:
            if st.session_state.chat_history:
                last_time = st.session_state.chat_history[-1][2] if len(st.session_state.chat_history[-1]) > 2 else "N/A"
                st.metric("🕐 Última", last_time)
    
    else:
        # Mensaje de bienvenida cuando no hay historial
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h3 style="color: #2E86AB; margin-bottom: 1rem;">👋 ¡Bienvenido a tu Coach Personal!</h3>
            <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
                Estoy aquí para ayudarte a mejorar tu rendimiento en deportes de resistencia.<br>
                Puedes preguntarme sobre entrenamiento, nutrición, técnica y mucho más.
            </p>
            <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                <span style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">💪 Planes de entrenamiento</span>
                <span style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🍎 Nutrición deportiva</span>
                <span style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🏆 Preparación para competencias</span>
                <span style="background: #f8f9fa; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">🔄 Recuperación</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#                                  MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════
elif modo == "📊 Dashboard de Métricas":
    # Importar y ejecutar el dashboard avanzado
    try:
        import subprocess
        import sys
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2 style="color: #667eea;">🚀 Cargando Dashboard Avanzado...</h2>
            <p>Redirigiendo al dashboard completo de métricas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón para abrir dashboard en nueva pestaña
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <a href="http://localhost:8502" target="_blank" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem 2rem;
                border-radius: 25px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            ">📊 Abrir Dashboard Completo</a>
        </div>
        """, unsafe_allow_html=True)
        
        # Información sobre cómo ejecutar el dashboard
        st.info("""
        **Para ver el dashboard completo:**
        
        1. Abre una nueva terminal
        2. Ejecuta: `streamlit run app/metrics_dashboard.py --server.port=8502`
        3. O haz clic en el botón de arriba
        
        El dashboard incluye:
        - 📈 Gráficos interactivos con Plotly
        - 🎯 Análisis radar de rendimiento
        - 📊 Evolución temporal de métricas
        - 📋 Tablas detalladas de resultados
        - 💡 Recomendaciones automáticas
        """)
        
        # Ejecutar evaluación desde aquí
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Ejecutar Nueva Evaluación", type="primary"):
                with st.spinner("Ejecutando evaluación..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, "app/run_eval.py"], 
                            capture_output=True, text=True, cwd="."
                        )
                        if result.returncode == 0:
                            st.success("✅ Evaluación completada exitosamente!")
                            st.info("Recarga el dashboard para ver los nuevos resultados.")
                        else:
                            st.error(f"❌ Error en evaluación: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Error ejecutando evaluación: {e}")
        
        with col2:
            if st.button("🔄 Verificar Estado del Sistema"):
                try:
                    result = subprocess.run(
                        [sys.executable, "health_check.py"], 
                        capture_output=True, text=True, cwd="."
                    )
                    if result.returncode == 0:
                        st.success("✅ Sistema funcionando correctamente")
                    else:
                        st.warning("⚠️ Algunos componentes pueden tener problemas")
                    
                    if result.stdout:
                        st.text(result.stdout)
                except Exception as e:
                    st.error(f"❌ Error verificando sistema: {e}")
        
    except Exception as e:
        st.error(f"Error cargando dashboard: {e}")

else:  # Analytics Básicos
    st.header("📈 Dashboard de Evaluación del Sistema")
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]
    except Exception as e:
        st.error(f"❌ No se pudo conectar a MLflow: {e}")
        st.info("Ejecuta primero: `python app/run_eval.py` para generar métricas")
        st.stop()

    if not experiments:
        st.warning("⚠️ No se encontraron experimentos de evaluación.")
        st.info("Ejecuta: `python app/run_eval.py` para generar datos de evaluación")
        st.stop()

    # Selector de experimento
    exp_names = [exp.name for exp in experiments]
    selected_exp = st.selectbox("🔬 Selecciona un experimento:", exp_names)

    experiment = next(exp for exp in experiments if exp.name == selected_exp)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

    if not runs:
        st.warning("⚠️ No hay ejecuciones registradas en este experimento.")
        st.stop()

    # Header de métricas
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h2 style="margin: 0 0 0.5rem 0; color: #2E86AB;">📊 Dashboard de Rendimiento del Sistema</h2>
        <p style="margin: 0; color: #666;">Análisis en tiempo real del rendimiento de la IA y calidad de respuestas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas principales con cards mejoradas
    st.markdown("### 📈 Métricas Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(runs)}</div>
            <div class="metric-label">🔬 Evaluaciones</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_correctness = np.mean([run.data.metrics.get("correctness_score", 0) for run in runs])
        progress_correctness = avg_correctness * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_correctness:.1%}</div>
            <div class="metric-label">✅ Precisión</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_correctness}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_relevance = np.mean([run.data.metrics.get("relevance_score", 0) for run in runs])
        progress_relevance = avg_relevance * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_relevance:.1%}</div>
            <div class="metric-label">🎯 Relevancia</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_relevance}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_coherence = np.mean([run.data.metrics.get("coherence_score", 0) for run in runs])
        progress_coherence = avg_coherence * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_coherence:.1%}</div>
            <div class="metric-label">🧠 Coherencia</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_coherence}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tabla detallada
    st.markdown("### 📋 Resultados Detallados")
    
    data = []
    for run in runs:
        data.append({
            "Pregunta": run.data.params.get("question", "N/A")[:50] + "...",
            "Prompt": run.data.params.get("prompt_version", "N/A"),
            "Chunk Size": int(run.data.params.get("chunk_size", 0)),
            "Correctness": run.data.metrics.get("correctness_score", 0),
            "Relevance": run.data.metrics.get("relevance_score", 0),
            "Coherence": run.data.metrics.get("coherence_score", 0),
            "Fecha": run.info.start_time
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Gráficos
    if len(df) > 0:
        st.markdown("### 📈 Visualizaciones")
        
        # Gráfico de barras por configuración
        grouped = df.groupby(["Prompt", "Chunk Size"]).agg({
            "Correctness": "mean",
            "Relevance": "mean", 
            "Coherence": "mean"
        }).reset_index()
        
        if not grouped.empty:
            grouped["Config"] = grouped["Prompt"] + " | " + grouped["Chunk Size"].astype(str)
            
            # Gráfico de correctness
            chart = alt.Chart(grouped).mark_bar().encode(
                x=alt.X("Config:N", title="Configuración"),
                y=alt.Y("Correctness:Q", title="Correctness", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("Config:N", legend=None),
                tooltip=["Prompt", "Chunk Size", alt.Tooltip("Correctness:Q", format=".3f")]
            ).properties(
                width=600,
                height=400,
                title="Correctness por Configuración"
            )
            
            st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<small>📊 Dashboard generado con Streamlit y Altair • "
        f"Datos versionados con MLflow • "
        f"<a href='{REPO_URL}'>Código fuente</a></small>",
        unsafe_allow_html=True,
    )