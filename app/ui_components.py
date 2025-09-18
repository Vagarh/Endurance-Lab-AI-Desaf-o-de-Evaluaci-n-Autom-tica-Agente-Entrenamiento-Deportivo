"""
Componentes reutilizables para la interfaz de usuario
"""
import streamlit as st
from datetime import datetime

def create_metric_card(title, value, icon, color="#2E86AB", progress=None):
    """Crea una tarjeta de métrica moderna"""
    progress_bar = ""
    if progress is not None:
        progress_bar = f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress}%; background: {color};"></div>
        </div>
        """
    
    return f"""
    <div class="metric-card" style="border-top-color: {color};">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 0.5rem;">{icon}</span>
            <div>
                <div class="metric-value" style="color: {color};">{value}</div>
                <div class="metric-label">{title}</div>
            </div>
        </div>
        {progress_bar}
    </div>
    """

def create_sport_card(sport, icon, description, color):
    """Crea una tarjeta deportiva"""
    return f"""
    <div class="sport-card">
        <div style="text-align: center;">
            <span class="sport-icon">{icon}</span>
            <h4 style="margin: 0.5rem 0; color: {color};">{sport}</h4>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">{description}</p>
        </div>
    </div>
    """

def create_chat_bubble(content, is_user=True, timestamp=None):
    """Crea una burbuja de chat"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if is_user:
        return f"""
        <div class="bubble-user">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>🧑‍💻 Tú</strong>
                <small style="opacity: 0.7;">{timestamp}</small>
            </div>
            <div>{content}</div>
        </div>
        """
    else:
        return f"""
        <div class="bubble-bot">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <strong>🤖 Coach Endurance</strong>
                <small style="opacity: 0.7;">{timestamp}</small>
            </div>
            <div>{content}</div>
        </div>
        """

def create_info_card(title, content, icon="ℹ️", color="#2E86AB"):
    """Crea una tarjeta informativa"""
    return f"""
    <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid {color};">
        <h4 style="margin: 0 0 1rem 0; color: {color}; display: flex; align-items: center;">
            <span style="margin-right: 0.5rem;">{icon}</span>
            {title}
        </h4>
        <div style="color: #666;">{content}</div>
    </div>
    """

def create_status_indicator(status="online", text="Sistema Activo"):
    """Crea un indicador de estado"""
    colors = {
        "online": "#28a745",
        "offline": "#dc3545", 
        "warning": "#ffc107"
    }
    
    return f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <div style="width: 12px; height: 12px; background: {colors.get(status, '#28a745')}; border-radius: 50%; margin-right: 8px; animation: pulse 2s infinite;"></div>
        <span style="color: #666; font-size: 0.9rem;">{text}</span>
    </div>
    """

def create_welcome_message():
    """Crea el mensaje de bienvenida"""
    return """
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
    """

def create_tips_carousel():
    """Crea un carrusel de tips"""
    tips = [
        {"icon": "💧", "title": "Hidratación", "content": "Mantente hidratado durante entrenamientos largos"},
        {"icon": "😴", "title": "Descanso", "content": "El descanso es tan importante como el entrenamiento"},
        {"icon": "📊", "title": "Registro", "content": "Registra tus entrenamientos para ver progreso"},
        {"icon": "🍎", "title": "Nutrición", "content": "La nutrición afecta directamente tu rendimiento"},
        {"icon": "🎯", "title": "Objetivos", "content": "Define metas específicas y medibles"},
        {"icon": "🔄", "title": "Progresión", "content": "Aumenta la intensidad gradualmente"}
    ]
    
    # Seleccionar tip basado en el día
    tip_del_dia = tips[datetime.now().day % len(tips)]
    
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; margin: 1rem 0;">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{tip_del_dia['icon']}</span>
            <div>
                <strong>Tip del día: {tip_del_dia['title']}</strong>
                <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.25rem;">{tip_del_dia['content']}</div>
            </div>
        </div>
    </div>
    """