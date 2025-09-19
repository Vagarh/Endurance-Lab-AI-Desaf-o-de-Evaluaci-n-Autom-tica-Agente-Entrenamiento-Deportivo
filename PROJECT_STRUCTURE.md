# 📁 Estructura del Proyecto - Endurance Lab AI v2.0.0

## 🏗️ Organización Mejorada

```
Endurance-Lab-AI/
├── 📱 app/                          # Aplicación principal
│   ├── __pycache__/
│   ├── prompts/                     # Plantillas de prompts
│   │   ├── v1_asistente_rrhh.txt
│   │   ├── v1_asistente_deporte.txt
│   │   └── v2_resumido_directo.txt
│   ├── main_interface.py            # Interfaz original
│   ├── main_interface_improved.py   # Interfaz mejorada ⭐
│   ├── rag_pipeline.py              # Pipeline RAG
│   ├── run_eval.py                  # Evaluación automática
│   ├── dashboard.py                 # Dashboard de métricas
│   └── ui_components.py             # Componentes UI reutilizables
│
├── 🛠️ scripts/                      # Scripts de automatización
│   ├── README.md
│   ├── start.py                     # Inicio rápido ⭐
│   ├── setup_project.py             # Configuración inicial
│   ├── restart_app.py               # Reinicio con mejoras
│   ├── create_vectorstore.py        # Creación de vectorstore
│   └── test_basic.py                # Tests básicos
│
├── ⚙️ utils/                        # Utilidades y configuración
│   └── config.py                    # Configuración centralizada ⭐
│
├── 📚 data/                         # Datos de entrenamiento
│   └── pdfs/                        # PDFs especializados (8 archivos)
│       ├── 1-Apunte-Entrenamiento-deportivo.pdf
│       ├── Bases-del-entrenamiento-deportivo-Tsvetan-Zhelyazkov.pdf
│       └── ... (6 más)
│
├── 🧪 tests/                        # Tests y evaluaciones
│   ├── eval_dataset.csv
│   ├── eval_dataset.json
│   └── test_run_eval.py
│
├── 🎨 assets/                       # Recursos estáticos
│   ├── README.md
│   ├── images/                      # Imágenes y logos
│   │   ├── ChatGPT Image 2 may 2025, 17_42_01.png
│   │   ├── ChatGPT Image 2 may 2025, 17_45_41.png
│   │   └── 6358552.jpg
│   └── docs/                        # Documentos estáticos
│       └── Informe_Escrito_Endurace_lab.pdf
│
├── 📖 docs/                         # Documentación
│   ├── README.md
│   ├── MEJORAS.md                   # Roadmap de mejoras
│   └── REVISION_COMPLETA.md         # Análisis completo
│
├── 🗄️ vectorstore/                  # Base de datos vectorial
│   ├── index.faiss
│   └── index.pkl
│
├── 📊 mlruns/                       # Experimentos MLflow
│   ├── 0/
│   └── 230736255057369722/
│
├── 📋 Archivos de configuración
│   ├── main.py                      # Punto de entrada principal ⭐
│   ├── .env                         # Variables de entorno
│   ├── .env.example                 # Plantilla de configuración
│   ├── requirements.txt             # Dependencias Python
│   ├── Dockerfile                   # Containerización
│   ├── README.md                    # Documentación principal
│   ├── LICENSE                      # Licencia MIT
│   └── PROJECT_STRUCTURE.md         # Este archivo
│
└── 🔧 Archivos del sistema
    ├── .git/                        # Control de versiones
    ├── .github/                     # GitHub Actions
    ├── .gitignore                   # Archivos ignorados
    └── __pycache__/                 # Cache de Python
```

## 🎯 Puntos de Entrada

### 🚀 Inicio Rápido
```bash
# Método recomendado
python main.py start

# Método directo
python scripts/start.py

# Método manual
streamlit run app/main_interface_improved.py
```

### ⚙️ Configuración
```bash
# Configuración inicial
python main.py setup

# Verificar configuración
python main.py config

# Tests básicos
python main.py test
```

## 📋 Beneficios de la Nueva Estructura

### ✅ **Organización Clara**
- Separación lógica por funcionalidad
- Fácil navegación y mantenimiento
- Estructura escalable

### ✅ **Mejor Mantenibilidad**
- Configuración centralizada
- Scripts organizados
- Documentación estructurada

### ✅ **Desarrollo Eficiente**
- Imports simplificados
- Reutilización de componentes
- Testing organizado

### ✅ **Despliegue Simplificado**
- Punto de entrada único
- Configuración clara
- Assets organizados

## 🔄 Migración desde v1.0.0

La nueva estructura mantiene **100% compatibilidad** con la versión anterior:

- ✅ Todos los archivos originales funcionan
- ✅ Scripts existentes siguen funcionando
- ✅ Configuración preservada
- ✅ Datos y modelos intactos

## 📈 Próximas Mejoras

- [ ] **CI/CD Pipeline** en `.github/workflows/`
- [ ] **Docker Compose** para desarrollo
- [ ] **API REST** en `app/api/`
- [ ] **Tests automatizados** en `tests/`
- [ ] **Documentación interactiva** en `docs/`

---

*Estructura actualizada: 18 de septiembre de 2025*  
*Versión: 2.0.0 - Organizada y Optimizada* 🏆