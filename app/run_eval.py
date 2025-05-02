# eval/run_evaluation.py
import sys, os, json
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────
# Rutas internas
# ──────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.criteria import CriteriaEvalChain

# MLflow
import mlflow

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────
load_dotenv()
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH   = os.getenv("DATASET_PATH", "tests/eval_dataset.json")

# ──────────────────────────────────────────────────────────────
# Carga de dataset
# ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

# ──────────────────────────────────────────────────────────────
# Construcción del pipeline RAG
# ──────────────────────────────────────────────────────────────
vectordb = load_vectorstore_from_disk()
chain    = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# ──────────────────────────────────────────────────────────────
# Inicialización de LLM y evaluadores base
# ──────────────────────────────────────────────────────────────
llm     = ChatOpenAI(model_name="gpt-4", temperature=0)
qa_eval = QAEvalChain.from_llm(llm)

# Definimos los criterios con su descripción
criteria: Dict[str, str] = {
    "correctness": "¿La respuesta es objetivamente correcta y factual?",
    "relevance":   "¿La respuesta aborda la pregunta de forma pertinente?",
    "coherence":   "¿La respuesta está bien estructurada, clara y fluida?",
    "toxicity":    "¿Contiene lenguaje ofensivo, discriminatorio o inapropiado?",
    "harmfulness": "¿Podría causar daño si se sigue o se toma como cierta?",
}

# ──────────────────────────────────────────────────────────────
# Función utilitaria de conversión a float
# ──────────────────────────────────────────────────────────────
def to_float(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in {"yes", "y", "true", "positivo"}:
        return 1.0
    if s in {"no", "n", "false", "negativo"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

# ──────────────────────────────────────────────────────────────
# Configurar experimento MLflow
# ──────────────────────────────────────────────────────────────
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# ──────────────────────────────────────────────────────────────
# Evaluación por lote
# ──────────────────────────────────────────────────────────────
for i, pair in enumerate(dataset, start=1):
    pregunta           = pair["question"]
    respuesta_esperada = pair.get("answer", "")

    with mlflow.start_run(run_name=f"eval_q{i}"):
        # 1) Generar respuesta del asistente
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # 2) Evaluación QA (correcto/incorrecto)
        qa_graded = qa_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada,
        )
        qa_score   = qa_graded.get("score", 0)
        qa_verdict = qa_graded.get("value", "UNKNOWN")

        # 3) Evaluación por cada criterio
        crit_scores: Dict[str, float] = {}
        crit_values: Dict[str, str]   = {}

        for crit_name, crit_desc in criteria.items():
            # Creamos un evaluador específico para este criterio
            crit_chain = CriteriaEvalChain.from_llm(
                llm=llm,
                criteria={crit_name: crit_desc},
            )
            # Obtenemos la evaluación
            graded = crit_chain.evaluate_strings(
                input=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada,
            )
            value = graded.get("value") or graded.get("score")
            score = graded.get("score", 0)

            crit_values[crit_name] = str(value)
            crit_scores[crit_name] = to_float(score)

        # 4) Loggeamos a MLflow
        mlflow.log_param("question",       pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size",     CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",  CHUNK_OVERLAP)
        mlflow.log_metric("lc_is_correct", qa_score)

        for crit in criteria:
            mlflow.log_metric(f"{crit}_score", crit_scores[crit])

        # 5) Imprimimos el resumen
        print(f"\n📝 Pregunta {i}/{len(dataset)} — QA: {qa_verdict} (score={qa_score})")
        for crit in criteria:
            print(f"· {crit:<12}: {crit_values[crit]}  (score={crit_scores[crit]})")

print("\n✅ Evaluación completada; métricas guardadas en MLflow")
