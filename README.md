═══════════════════════════════════════════════════════
🏆  ENDURANCE LAB AI — Asistente Virtual Deportivo
═══════════════════════════════════════════════════════

💡 **Asistente IA especializado en entrenamiento de resistencia**  
   🚴‍♂️ Ciclismo · 🏃‍♂️ Running · 🏊‍♂️ Natación · 🏆 Triatlón

📌 **Proyecto Final de los cursos:**
   • Procesamiento de Lenguaje Natural  
   • Experiencias en Inteligencia de Negocios  

👤 **Autor:** Juan Felipe Cardona Arango  
📅 **Fecha:** Mayo 2025 - Actualizado Septiembre 2025  
� ** Versión:** 2.0.0 - Estructura Profesional Organizada  

👨‍🏫 **Docentes:**  
   • Juan David Martínez Vargas  
   • Ana María López Moreno  
   • Edwin Nelson Montoya Múnera  

🌐 **Demo en Vivo:** https://gwzq2khrmuwgkzbhavrmpb.streamlit.app/  
📚 **Repositorio:** [GitHub](https://github.com/Vagarh/Endurance-Lab-AI)


-------------------------------------------------------
📖 DESCRIPCIÓN GENERAL
-------------------------------------------------------

**Endurance Lab AI** es un asistente virtual inteligente especializado en entrenamiento de resistencia, desarrollado con tecnologías de vanguardia:

### 🎯 **Características Principales**
✅ **IA Conversacional** - Chat inteligente con GPT-4o  
✅ **RAG Especializado** - Base de conocimiento deportiva  
✅ **Evaluación Automática** - Métricas de calidad con MLflow  
✅ **Dashboard Interactivo** - Visualizaciones avanzadas  
✅ **Interfaz Moderna** - UI mejorada con Streamlit  
✅ **Estructura Profesional** - Código organizado y escalable  

### 🏗️ **Arquitectura Técnica**
- **Backend:** Python 3.10+ con LangChain
- **Frontend:** Streamlit con UI personalizada
- **IA:** OpenAI GPT-4o + embeddings
- **Base de Datos:** FAISS vectorstore
- **Monitoreo:** MLflow para métricas
- **Despliegue:** Docker + Nginx

-------------------------------------------------------
⚙️ REQUISITOS DEL SISTEMA
-------------------------------------------------------
- **Python:** 3.10 o superior
- **Memoria:** 4GB RAM mínimo (8GB recomendado)
- **Espacio:** 2GB libres
- **API Key:** OpenAI (configurar en `.env`)
- **SO:** Windows, macOS, Linux

-------------------------------------------------------
🏗️ ESTRUCTURA DEL PROYECTO (v2.0.0)
-------------------------------------------------------

```
🏆 Endurance-Lab-AI/
├── 📱 app/                    # Aplicación principal
│   ├── main_interface_improved.py  # Interfaz mejorada ⭐
│   ├── rag_pipeline.py            # Pipeline RAG
│   ├── run_eval.py                # Evaluación automática
│   ├── metrics_dashboard.py       # Dashboard avanzado
│   ├── ui_components.py           # Componentes reutilizables
│   └── prompts/                   # Plantillas especializadas
├── 🛠️ scripts/               # Automatización
│   ├── start.py                   # Inicio rápido
│   ├── setup_project.py           # Configuración inicial
│   ├── cleanup.py                 # Limpieza del proyecto
│   └── test_basic.py              # Tests básicos
├── ⚙️ utils/                 # Configuración
│   └── config.py                  # Configuración centralizada
├── 📚 data/pdfs/             # Base de conocimiento (8 PDFs)
├── 🧪 tests/                 # Tests y evaluaciones
├── 🎨 assets/                # Recursos estáticos
│   ├── images/                    # Logos e imágenes
│   └── docs/                      # Documentos PDF
├── 📖 docs/                  # Documentación
│   ├── MEJORAS.md                 # Roadmap de mejoras
│   └── REVISION_COMPLETA.md       # Análisis técnico
├── 🗄️ vectorstore/           # Base de datos vectorial
├── 📊 mlruns/                # Experimentos MLflow
└── 📋 main.py                # Punto de entrada principal ⭐
```

📋 **Documentación Completa:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

-------------------------------------------------------
🎯 FUNCIONALIDADES PRINCIPALES
-------------------------------------------------------

### 🤖 **Asistente IA Conversacional**
- **Chat Inteligente** con especialización deportiva
- **Personalización** por disciplina y nivel de experiencia
- **Contexto Adaptativo** según objetivos del usuario
- **Respuestas Basadas en Evidencia** científica

### 📊 **Dashboard de Métricas Avanzado**
- **Visualizaciones Interactivas** con gráficos radar
- **Evaluación Automática** de calidad de respuestas
- **Métricas de Rendimiento** (Correctness, Relevance, Coherence)
- **Análisis Temporal** de mejoras del sistema

### 🏃‍♂️ **Especialización Deportiva**
- **🚴‍♂️ Ciclismo:** FTP, zonas de potencia, biomecánica
- **🏃‍♂️ Running:** Pace, técnica, prevención de lesiones
- **🏊‍♂️ Natación:** Técnica de brazada, entrenamiento en piscina
- **🏆 Triatlón:** Transiciones, estrategias combinadas

### 📚 **Base de Conocimiento Especializada**
- **8 PDFs Técnicos** sobre entrenamiento deportivo
- **Vectorstore FAISS** para búsqueda semántica
- **Prompts Especializados** por disciplina
- **Evaluación Continua** de calidad de contenido

-------------------------------------------------------
🔬 SISTEMA DE EVALUACIÓN AUTOMÁTICA
-------------------------------------------------------

### 📊 **Métricas de Calidad Evaluadas**
- **✅ Correctness** - Precisión factual de las respuestas
- **🎯 Relevance** - Pertinencia respecto a la consulta
- **🧠 Coherence** - Claridad y estructura del contenido
- **🛡️ Safety** - Ausencia de toxicidad y contenido dañino
- **💬 Clarity** - Comprensibilidad para deportistas

### 🛠️ **Tecnologías de Evaluación**
- **Framework:** LangChain para evaluación automática
- **Tracking:** MLflow para seguimiento de experimentos
- **Métricas:** Scores numéricos (0-1) + explicaciones textuales
- **Comparación:** Análisis entre diferentes configuraciones

### � **-Resultados de Rendimiento**
```
Configuración Óptima: Chunk 256 + Prompt Deportivo
┌─────────────┬──────────┬───────────┬───────────┐
│ Métrica     │ Score    │ Benchmark │ Estado    │
├─────────────┼──────────┼───────────┼───────────┤
│ Correctness │ 92%      │ >80%      │ ✅ Excelente │
│ Relevance   │ 90%      │ >75%      │ ✅ Excelente │
│ Coherence   │ 88%      │ >70%      │ ✅ Excelente │
│ Safety      │ 100%     │ 100%      │ ✅ Perfecto  │
│ Clarity     │ 85%      │ >70%      │ ✅ Excelente │
└─────────────┴──────────┴───────────┴───────────┘
```

### 🎯 **Comandos de Evaluación**
```bash
# Ejecutar evaluación completa
python app/run_eval.py

# Dashboard de métricas avanzado
python run_metrics_dashboard.py

# Ver resultados en MLflow
mlflow ui
```

-------------------------------------------------------
� INICOIO RÁPIDO - ESTRUCTURA v2.0.0
-------------------------------------------------------

### ⚡ **Método Recomendado (3 pasos)**
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key de OpenAI
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY

# 3. Iniciar aplicación
python main.py start
```

### 🎯 **Comandos Principales**
```bash
python main.py start          # 🚀 Iniciar aplicación
python main.py setup          # ⚙️ Configuración inicial
python main.py config         # 📋 Ver configuración
python main.py test           # 🧪 Ejecutar tests
python main.py help           # ❓ Mostrar ayuda
```

### 🔧 **Métodos Alternativos**
```bash
# Inicio directo con scripts
python scripts/start.py

# Interfaz mejorada directamente
streamlit run app/main_interface_improved.py

# Dashboard de métricas avanzado
python run_metrics_dashboard.py
```

### 🌐 **Acceso a la Aplicación**
- **Desarrollo:** http://localhost:8501
- **Dashboard Métricas:** http://localhost:8502
- **Producción:** Configurar según `DEPLOYMENT.md`

-------------------------------------------------------
🐳 DESPLIEGUE EN PRODUCCIÓN
-------------------------------------------------------

### 🚀 **Despliegue Automatizado**
```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh && ./deploy.sh
```

### 🐳 **Docker (Recomendado para Producción)**
```bash
# Construcción y despliegue
docker-compose build
docker-compose up -d

# Verificar estado del sistema
python health_check.py

# Ver logs en tiempo real
docker-compose logs -f
```

### 🔧 **Comandos de Mantenimiento**
```bash
# Limpieza del proyecto
python scripts/cleanup.py

# Verificar configuración completa
python main.py config

# Ejecutar tests de integridad
python scripts/test_basic.py

# Monitoreo de salud del sistema
python health_check.py
```

### 🌐 **URLs de Acceso**
- **🖥️ Aplicación Principal:** http://localhost:8501
- **📊 Dashboard Métricas:** http://localhost:8502  
- **🔍 MLflow UI:** http://localhost:5000
- **🏥 Health Check:** http://localhost:8501/health

📖 **Guía Completa:** Ver [DEPLOYMENT.md](DEPLOYMENT.md) para instrucciones detalladas

-------------------------------------------------------
� DACSHBOARD DE MÉTRICAS AVANZADO
-------------------------------------------------------

### 🎯 **Dashboard Interactivo Completo**

**Acceso Rápido:**
```bash
# Método recomendado
python run_metrics_dashboard.py

# Windows (script batch)
run_metrics_dashboard.bat

# Comando directo
streamlit run app/metrics_dashboard.py --server.port=8502
```

### ✨ **Características del Dashboard**
- **📈 Gráficos Radar** - Rendimiento vs objetivos
- **📊 Evolución Temporal** - Métricas a lo largo del tiempo  
- **🔍 Análisis Detallado** - Por configuración y prompt
- **� T*ablas Interactivas** - Resultados filtrados
- **🚀 Ejecución en Vivo** - Evaluaciones desde la interfaz
- **💡 Recomendaciones** - Sugerencias automáticas de mejora

### 📈 **Interpretación de Resultados**
```
🟢 Excelente (80-100%) → Listo para producción
🟡 Bueno (60-79%)      → Margen de mejora  
🔴 Necesita Mejora (<60%) → Requiere optimización
```

🌐 **Acceso Dashboard:** http://localhost:8502

-------------------------------------------------------
🛠️ DESARROLLO Y CONTRIBUCIÓN
-------------------------------------------------------

### 🔧 **Para Desarrolladores**
```bash
# Clonar repositorio
git clone https://github.com/Vagarh/Endurance-Lab-AI.git
cd Endurance-Lab-AI

# Configurar entorno de desarrollo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias de desarrollo
pip install -r requirements.txt

# Configurar pre-commit hooks
pre-commit install
```

### 📚 **Documentación Técnica**
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Estructura detallada
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Guía de despliegue
- **[docs/MEJORAS.md](docs/MEJORAS.md)** - Roadmap de mejoras
- **[docs/REVISION_COMPLETA.md](docs/REVISION_COMPLETA.md)** - Análisis técnico

### 🤝 **Contribuir al Proyecto**
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

-------------------------------------------------------
📄 LICENCIA Y CRÉDITOS
-------------------------------------------------------

### 📜 **Licencia**
Este proyecto está licenciado bajo los términos de la **Licencia MIT**.  
Ver [LICENSE](LICENSE) para más información.

### 🙏 **Agradecimientos**
- **OpenAI** por GPT-4o y embeddings
- **LangChain** por el framework RAG
- **Streamlit** por la interfaz web
- **MLflow** por el tracking de experimentos
- **Comunidad Open Source** por las librerías utilizadas

### 📞 **Contacto y Soporte**
- **📧 Email:** [Contacto del autor]
- **🐛 Issues:** [GitHub Issues](https://github.com/Vagarh/Endurance-Lab-AI/issues)
- **💬 Discusiones:** [GitHub Discussions](https://github.com/Vagarh/Endurance-Lab-AI/discussions)

═══════════════════════════════════════════════════════

### 🏆 **Estado del Proyecto: LISTO PARA PRODUCCIÓN**

✅ **Estructura Profesional Organizada**  
✅ **Interfaz Moderna y Funcional**  
✅ **Sistema de Evaluación Robusto**  
✅ **Documentación Completa**  
✅ **Despliegue Automatizado**  

**🚀 ¡Comienza a usar Endurance Lab AI ahora!**

═══════════════════════════════════════════════════════#   T e s t   t r i g g e r  
 