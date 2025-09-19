"""
Dashboard de Métricas Avanzado para Endurance Lab AI
Visualización completa de resultados de evaluación con MLflow
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Configuración de página
st.set_page_config(
    page_title="📊 Métricas - Endurance Lab AI",
    layout="wide",
    page_icon="📈"
)

# CSS personalizado para el dashboard
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-excellent { color: #28a745; }
    .status-good { color: #ffc107; }
    .status-needs-improvement { color: #dc3545; }
    
    .evaluation-summary {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_status_color(score):
    """Determina el color según el score"""
    if score >= 0.8:
        return "status-excellent"
    elif score >= 0.6:
        return "status-good"
    else:
        return "status-needs-improvement"

def get_status_emoji(score):
    """Determina el emoji según el score"""
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    else:
        return "🔴"

@st.cache_data
def load_mlflow_data():
    """Carga datos de MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = [exp for exp in client.search_experiments() 
                      if exp.name.startswith("eval_")]
        
        if not experiments:
            return None, []
        
        # Obtener el experimento más reciente
        latest_exp = max(experiments, key=lambda x: x.creation_time)
        runs = client.search_runs(
            experiment_ids=[latest_exp.experiment_id],
            order_by=["start_time DESC"]
        )
        
        return latest_exp, runs
    except Exception as e:
        st.error(f"Error cargando datos de MLflow: {e}")
        return None, []

def create_metrics_summary(runs):
    """Crea resumen de métricas"""
    if not runs:
        return {}
    
    metrics = ['correctness_score', 'relevance_score', 'coherence_score', 
               'toxicity_score', 'harmfulness_score']
    
    summary = {}
    for metric in metrics:
        values = [run.data.metrics.get(metric, 0) for run in runs]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len([v for v in values if v > 0])
        }
    
    return summary

def create_radar_chart(summary):
    """Crea gráfico radar de métricas"""
    metrics = ['Correctness', 'Relevance', 'Coherence', 'Safety (1-Toxicity)', 'Safety (1-Harmfulness)']
    values = [
        summary.get('correctness_score', {}).get('mean', 0),
        summary.get('relevance_score', {}).get('mean', 0),
        summary.get('coherence_score', {}).get('mean', 0),
        1 - summary.get('toxicity_score', {}).get('mean', 0),  # Invertir para que más alto = mejor
        1 - summary.get('harmfulness_score', {}).get('mean', 0)  # Invertir para que más alto = mejor
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Rendimiento Actual',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Línea de referencia (objetivo)
    target_values = [0.85, 0.90, 0.80, 0.95, 0.95]
    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=metrics,
        fill='toself',
        name='Objetivo',
        line_color='rgb(40, 167, 69)',
        fillcolor='rgba(40, 167, 69, 0.1)',
        line_dash='dash'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Rendimiento del Sistema vs Objetivos",
        height=500
    )
    
    return fig

def create_timeline_chart(runs):
    """Crea gráfico de evolución temporal"""
    if len(runs) < 2:
        return None
    
    data = []
    for run in reversed(runs):  # Orden cronológico
        timestamp = datetime.fromtimestamp(run.info.start_time / 1000)
        data.append({
            'timestamp': timestamp,
            'correctness': run.data.metrics.get('correctness_score', 0),
            'relevance': run.data.metrics.get('relevance_score', 0),
            'coherence': run.data.metrics.get('coherence_score', 0),
            'run_id': run.info.run_id[:8]
        })
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Evolución de Métricas en el Tiempo',)
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['correctness'], 
                  mode='lines+markers', name='Correctness',
                  line=dict(color='#667eea', width=3))
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['relevance'], 
                  mode='lines+markers', name='Relevance',
                  line=dict(color='#764ba2', width=3))
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['coherence'], 
                  mode='lines+markers', name='Coherence',
                  line=dict(color='#f093fb', width=3))
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Tiempo",
        yaxis_title="Score (0-1)",
        hovermode='x unified'
    )
    
    return fig

def create_detailed_table(runs):
    """Crea tabla detallada de resultados"""
    data = []
    for i, run in enumerate(runs):
        question = run.data.params.get('question', 'N/A')
        if len(question) > 60:
            question = question[:60] + "..."
        
        data.append({
            'Run': f"#{i+1}",
            'Pregunta': question,
            'Prompt': run.data.params.get('prompt_version', 'N/A'),
            'Chunk Size': run.data.params.get('chunk_size', 'N/A'),
            'Correctness': f"{run.data.metrics.get('correctness_score', 0):.2f}",
            'Relevance': f"{run.data.metrics.get('relevance_score', 0):.2f}",
            'Coherence': f"{run.data.metrics.get('coherence_score', 0):.2f}",
            'Toxicity': f"{run.data.metrics.get('toxicity_score', 0):.2f}",
            'Harmfulness': f"{run.data.metrics.get('harmfulness_score', 0):.2f}",
            'Fecha': datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M")
        })
    
    return pd.DataFrame(data)

# Header principal
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #667eea; font-size: 3rem; margin-bottom: 0.5rem;">📊 Dashboard de Métricas</h1>
    <p style="color: #666; font-size: 1.2rem;">Análisis de Rendimiento del Sistema Endurance Lab AI</p>
</div>
""", unsafe_allow_html=True)

# Cargar datos
experiment, runs = load_mlflow_data()

if not experiment or not runs:
    st.warning("⚠️ No se encontraron datos de evaluación.")
    st.info("Ejecuta `python app/run_eval.py` para generar métricas.")
    
    # Botón para ejecutar evaluación
    if st.button("🚀 Ejecutar Evaluación Ahora", type="primary"):
        with st.spinner("Ejecutando evaluación..."):
            import subprocess
            try:
                result = subprocess.run(["python", "app/run_eval.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Evaluación completada. Recarga la página para ver resultados.")
                    st.rerun()
                else:
                    st.error(f"❌ Error en evaluación: {result.stderr}")
            except Exception as e:
                st.error(f"❌ Error ejecutando evaluación: {e}")
    st.stop()

# Crear resumen de métricas
summary = create_metrics_summary(runs)

# Métricas principales
st.markdown("## 📈 Métricas Principales")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    correctness_avg = summary.get('correctness_score', {}).get('mean', 0)
    status_class = get_status_color(correctness_avg)
    emoji = get_status_emoji(correctness_avg)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Correctness {emoji}</div>
        <div class="metric-value">{correctness_avg:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    relevance_avg = summary.get('relevance_score', {}).get('mean', 0)
    emoji = get_status_emoji(relevance_avg)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Relevance {emoji}</div>
        <div class="metric-value">{relevance_avg:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    coherence_avg = summary.get('coherence_score', {}).get('mean', 0)
    emoji = get_status_emoji(coherence_avg)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Coherence {emoji}</div>
        <div class="metric-value">{coherence_avg:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    toxicity_avg = summary.get('toxicity_score', {}).get('mean', 0)
    safety_score = 1 - toxicity_avg  # Invertir para que más alto = mejor
    emoji = get_status_emoji(safety_score)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Safety {emoji}</div>
        <div class="metric-value">{safety_score:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Runs</div>
        <div class="metric-value">{len(runs)}</div>
    </div>
    """, unsafe_allow_html=True)

# Gráfico radar
st.markdown("## 🎯 Análisis de Rendimiento")
col1, col2 = st.columns([2, 1])

with col1:
    radar_fig = create_radar_chart(summary)
    st.plotly_chart(radar_fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Interpretación")
    
    overall_score = np.mean([
        correctness_avg, relevance_avg, coherence_avg, 
        1-summary.get('toxicity_score', {}).get('mean', 0),
        1-summary.get('harmfulness_score', {}).get('mean', 0)
    ])
    
    if overall_score >= 0.8:
        st.success(f"🟢 **Excelente** ({overall_score:.1%})")
        st.write("El sistema está funcionando de manera óptima.")
    elif overall_score >= 0.6:
        st.warning(f"🟡 **Bueno** ({overall_score:.1%})")
        st.write("El sistema funciona bien pero tiene margen de mejora.")
    else:
        st.error(f"🔴 **Necesita Mejora** ({overall_score:.1%})")
        st.write("El sistema requiere optimización significativa.")
    
    st.markdown("### 🎯 Objetivos")
    st.write("- Correctness: ≥85%")
    st.write("- Relevance: ≥90%") 
    st.write("- Coherence: ≥80%")
    st.write("- Safety: ≥95%")

# Evolución temporal
if len(runs) > 1:
    st.markdown("## 📈 Evolución Temporal")
    timeline_fig = create_timeline_chart(runs)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)

# Tabla detallada
st.markdown("## 📋 Resultados Detallados")
detailed_df = create_detailed_table(runs)
st.dataframe(detailed_df, use_container_width=True)

# Análisis por configuración
st.markdown("## ⚙️ Análisis por Configuración")

config_analysis = {}
for run in runs:
    prompt = run.data.params.get('prompt_version', 'unknown')
    chunk_size = run.data.params.get('chunk_size', 'unknown')
    config_key = f"{prompt} | {chunk_size}"
    
    if config_key not in config_analysis:
        config_analysis[config_key] = {
            'correctness': [],
            'relevance': [],
            'coherence': []
        }
    
    config_analysis[config_key]['correctness'].append(
        run.data.metrics.get('correctness_score', 0)
    )
    config_analysis[config_key]['relevance'].append(
        run.data.metrics.get('relevance_score', 0)
    )
    config_analysis[config_key]['coherence'].append(
        run.data.metrics.get('coherence_score', 0)
    )

if config_analysis:
    config_df = []
    for config, metrics in config_analysis.items():
        config_df.append({
            'Configuración': config,
            'Correctness': f"{np.mean(metrics['correctness']):.2f}",
            'Relevance': f"{np.mean(metrics['relevance']):.2f}",
            'Coherence': f"{np.mean(metrics['coherence']):.2f}",
            'Runs': len(metrics['correctness'])
        })
    
    config_df = pd.DataFrame(config_df)
    st.dataframe(config_df, use_container_width=True)

# Recomendaciones
st.markdown("## 💡 Recomendaciones")

recommendations = []

if correctness_avg < 0.7:
    recommendations.append("🔧 **Mejorar Correctness**: Revisar y actualizar la base de conocimientos con información más precisa.")

if relevance_avg < 0.8:
    recommendations.append("🎯 **Mejorar Relevance**: Optimizar los prompts para respuestas más directas y pertinentes.")

if coherence_avg < 0.7:
    recommendations.append("📝 **Mejorar Coherence**: Ajustar la estructura de respuestas para mayor claridad.")

if summary.get('toxicity_score', {}).get('mean', 0) > 0.1:
    recommendations.append("⚠️ **Reducir Toxicity**: Implementar filtros adicionales de contenido.")

if not recommendations:
    recommendations.append("✅ **Excelente rendimiento**: El sistema está funcionando dentro de los parámetros esperados.")

for rec in recommendations:
    st.markdown(f"- {rec}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        📊 Dashboard generado automáticamente • 
        Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • 
        Experimento: {experiment.name if experiment else 'N/A'}
    </small>
</div>
""", unsafe_allow_html=True)