═══════════════════════════════════════════════════════
🏆  ENDURANCE LAB AI — Asistente Virtual Deportivo
═══════════════════════════════════════════════════════

💡 Asistente especializado en entrenamiento de resistencia:  
   ciclismo · running · triatlón · natación

📌 Proyecto Final de los cursos:
   • Procesamiento de Lenguaje Natural  
   • Experiencias en Inteligencia de Negocios  

👤 Autor: Juan Felipe Cardona Arango  
📅 Fecha: Mayo 2025  
👨‍🏫 Docentes:  
   • Juan David Martínez Vargas  
   • Ana María López Moreno  
   • Edwin Nelson Montoya Múnera  

-------------------------------------------------------
📖 DESCRIPCIÓN GENERAL
-------------------------------------------------------

Este repositorio contiene el desarrollo completo del asistente **Endurance Lab AI**:
✔ Personalización del dominio y prompts  
✔ Evaluación automática con LangChain + MLflow  
✔ Dataset de pruebas y análisis  
✔ Dashboard interactivo  
✔ Reflexiones finales y criterio de claridad para usuarios deportistas

-------------------------------------------------------
⚙️ REQUISITOS BÁSICOS
-------------------------------------------------------
- Python 3.10 o superior
- Copia `.env.example` a `.env` y agrega tu `OPENAI_API_KEY`

-------------------------------------------------------
🗂️ ESTRUCTURA DEL CONTENIDO
-------------------------------------------------------

1. ▶ Parte 1: Personalización
2. ▶ Parte 2: Evaluación Automática
3. ▶ Parte 3: Reto Investigador
4. ▶ Parte 4: Dashboard de Métricas
5. ▶ Parte 5: Presentación y Reflexión
6. ▶ BONUS: Evaluación de Claridad
7. ▶ Cómo Ejecutar el Proyecto
8. ▶ Licencia

-------------------------------------------------------
📁 PARTE 1: PERSONALIZACIÓN
-------------------------------------------------------

🔹 Dominio Temático:
   • Entrenamiento deportivo de resistencia
   • Foco en ciclismo, triatlón, natación y running

🔹 Documentos Internos Sustituidos:
   • planes_entrenamiento.pdf
   • historiales_rendimiento.pdf
   • guias_nutricion.pdf
   • revisiones_bibliograficas.pdf

🔹 Prompts:
   ▫ Principal → Respuestas completas, con validación de contexto y cita textual
   ▫ Secundario → Respuestas breves y directas, solo si hay contexto suficiente

🔹 Dataset de Pruebas:
   • Archivo: eval_dataset.csv  
   • Incluye casos reales y extremos

-------------------------------------------------------
🤖 PARTE 2: EVALUACIÓN AUTOMÁTICA
-------------------------------------------------------

⚙ Script principal: run_eval.py  
🧠 Framework: LangChain  
📊 Seguimiento: MLflow

Criterios evaluados automáticamente:
• Correctness  
• Relevance  
• Coherence  
• Toxicity  
• Harmfulness

-------------------------------------------------------
🔬 PARTE 3: RETO INVESTIGADOR
-------------------------------------------------------

Se añadieron criterios avanzados personalizados:
• *_score → Métrica numérica  
• *_reasoning → Explicación textual

-------------------------------------------------------
📊 PARTE 4: DASHBOARD DE MÉTRICAS
-------------------------------------------------------

🗂 Archivos clave:
   • dashboard.py → Gráficas por criterio  
   • app/main_interface.py → Interfaz visual y filtros

-------------------------------------------------------
📈 PARTE 5: PRESENTACIÓN Y REFLEXIÓN
-------------------------------------------------------

Evaluación comparativa:

 Configuración            | Correct | Relevant | Coherent | Toxic | Harmful
 -------------------------|---------|----------|----------|-------|---------
 Chunk = 512 · Prompt A   |  0.87   |   0.85   |   0.82   | 0.00  |  0.00
 Chunk = 256 · Prompt B   |  0.92   |   0.90   |   0.88   | 0.00  |  0.00

✅ Mejor combinación: Chunk 256 + Prompt B  
⚠️ Hallazgos: Incoherencia leve y formato inconsistente en chunks largos

-------------------------------------------------------
✨ BONUS: CRITERIO DE CLARIDAD
-------------------------------------------------------

Evalúa si la respuesta es comprensible para un deportista (fluidez, jerga, estructura).  
Resultados guardados como:  
• clarity_score (valor numérico)  
• clarity_reasoning (explicación registrada en MLflow)

-------------------------------------------------------
🛠️ CÓMO EJECUTAR EL PROYECTO
-------------------------------------------------------

### 🚀 Despliegue en Producción (Recomendado)
```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh
./deploy.sh
```

### 🔧 Desarrollo Local
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key de OpenAI en .env
# (Edita .env y reemplaza 'tu_clave_aqui' con tu API key)

# 3. Ejecutar script de inicio automático
python start.py
```

### 📋 Pasos Manuales (Alternativo)
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY

# 3. Crear vectorstore (solo la primera vez)
python create_vectorstore.py

# 4. Ejecutar tests básicos (opcional)
python test_basic.py

# 5. Lanzar aplicación
streamlit run app/main_interface_improved.py
```

### 🐳 Docker (Producción)
```bash
# Construcción y despliegue
docker-compose build
docker-compose up -d

# Verificar estado
python health_check.py

# Ver logs
docker-compose logs -f
```

### 🔧 Comandos Adicionales
```bash
# Ejecutar evaluación automática
python app/run_eval.py

# Configuración inicial completa
python setup_project.py

# Verificar configuración
python -c "from config import validate_config; print(validate_config())"
```

🌐 **Acceso:** 
- Desarrollo: http://localhost:8501
- Producción: http://localhost (via Nginx)

-------------------------------------------------------
📄 LICENCIA
-------------------------------------------------------

Este proyecto está licenciado bajo los términos de la Licencia MIT.  
Revisa el archivo LICENSE para más información.

═══════════════════════════════════════════════════════


-------------------------------------------------------
🚀 PREPARACIÓN PARA PRODUCCIÓN COMPLETADA
-------------------------------------------------------

✅ **Tu proyecto está ahora completamente listo para producción con:**

### 🐳 **Containerización**
- Docker optimizado para producción
- Docker Compose para orquestación
- Health checks automáticos
- Imagen multi-stage optimizada

### 🔒 **Seguridad**
- Nginx como proxy reverso
- Rate limiting (10 req/s por IP)
- Headers de seguridad
- Usuario no-root en contenedor
- Variables de entorno seguras

### 📊 **Monitoreo**
- Health check endpoint
- Logs estructurados
- Métricas de sistema
- Script de verificación automática

### 🚀 **Despliegue Automatizado**
- Script de despliegue para Windows (`deploy.bat`)
- Script de despliegue para Linux/Mac (`deploy.sh`)
- Configuración de producción lista
- Backup y recovery procedures

### 📈 **Escalabilidad**
- Configuración para múltiples instancias
- Load balancing preparado
- Recursos optimizados
- Preparado para cloud deployment

**📖 Ver `DEPLOYMENT.md` para guía completa de despliegue**

-------------------------------------------------------