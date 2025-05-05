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

1️⃣ Instalar dependencias:
   pip install -r requirements.txt

2️⃣ Ejecutar la evaluación automática:
   python run_eval.py --dataset eval_dataset.csv

3️⃣ Lanzar la aplicación Streamlit:
   streamlit run app/main_interface.py

🌐 Luego, accede en tu navegador a:
   http://localhost:8501

-------------------------------------------------------
📄 LICENCIA
-------------------------------------------------------

Este proyecto está licenciado bajo los términos de la Licencia MIT.  
Revisa el archivo LICENSE para más información.

═══════════════════════════════════════════════════════

