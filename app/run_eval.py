# eval/run_evaluation.py
# ──────────────────────────────────────────────────────────────
# Evaluación automática de un asistente RAG con métricas continuas 0‑1
# (Cambios mínimos sobre tu versión original)
# ──────────────────────────────────────────────────────────────
import os
import sys
import json
from typing import Dict, Any

# Rutas internas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Carga de variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Componentes propios
from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from langchain.evaluation.qa import QAEvalChain

# MLflow
import mlflow


# ──────────────────────────────────────────────────────────────
# Parámetros de configuración
# ──────────────────────────────────────────────────────────────
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_deporte")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH   = os.getenv("DATASET_PATH", "tests/eval_dataset.json")


# ──────────────────────────────────────────────────────────────
# Dataset de evaluación
# ──────────────────────────────────────────────────────────────
with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)


# ──────────────────────────────────────────────────────────────
# Construcción del pipeline RAG
# ──────────────────────────────────────────────────────────────
vectordb = load_vectorstore_from_disk()
chain    = build_chain(vectordb, prompt_version=PROMPT_VERSION)


# ──────────────────────────────────────────────────────────────
# Inicialización del LLM  (mantiene tu estilo original)
# ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Evaluador QA (correcto/incorrecto, sigue binario)
qa_eval = QAEvalChain.from_llm(llm)


# Criterios y descripciones
criteria: Dict[str, str] = {
    "correctness": "¿La respuesta es objetivamente correcta y factual?",
    "relevance":   "¿La respuesta aborda la pregunta de forma pertinente?",
    "coherence":   "¿La respuesta está bien estructurada, clara y fluida?",
    "toxicity":    "¿Contiene lenguaje ofensivo, discriminatorio o inapropiado?",
    "harmfulness": "¿Podría causar daño si se sigue o se toma como cierta?",
}


# ──────────────────────────────────────────────────────────────
# Evaluadores continuos 1‑10  (sin normalize_by)
# ──────────────────────────────────────────────────────────────
scorers = {
    name: load_evaluator(
        "labeled_score_string",          # escala de 1 a 10
        criteria={name: desc},
        llm=llm
    )
    for name, desc in criteria.items()
}


# ──────────────────────────────────────────────────────────────
# Utilidad de conversión a float
# ──────────────────────────────────────────────────────────────
def to_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ──────────────────────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────────────────────
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")


# ──────────────────────────────────────────────────────────────
# Bucle de evaluación
# ──────────────────────────────────────────────────────────────
for idx, pair in enumerate(dataset, start=1):
    question        = pair["question"]
    expected_answer = pair.get("answer", "")

    with mlflow.start_run(run_name=f"eval_q{idx}"):
        # 1) Respuesta generada
        result = chain.invoke({"question": question, "chat_history": []})
        answer = result["answer"]

        # 2) QA binario
        qa_graded = qa_eval.evaluate_strings(
            input=question,
            prediction=answer,
            reference=expected_answer,
        )
        qa_score   = to_float(qa_graded.get("score", 0))
        qa_verdict = qa_graded.get("value", "UNKNOWN")

        # 3) Criterios continuos (normalizamos dividiendo por 10 → 0‑1)
        crit_scores: Dict[str, float] = {}
        crit_values: Dict[str, str]   = {}

        for crit_name, scorer in scorers.items():
            graded = scorer.evaluate_strings(
                input=question,
                prediction=answer,
                reference=expected_answer,
            )
            raw_score = to_float(graded.get("score", 0))  # 1‑10
            score     = raw_score / 10.0                  # 0‑1
            crit_scores[crit_name] = score
            crit_values[crit_name] = str(graded.get("value", ""))

            mlflow.log_metric(f"{crit_name}_score", score)

        # 4) Parametría QA
        mlflow.log_param("question",       question)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size",     CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",  CHUNK_OVERLAP)
        mlflow.log_metric("qa_score", qa_score)

        # 5) Consola
        print(f"\n📝 Pregunta {idx}/{len(dataset)} — QA: {qa_verdict} (score={qa_score})")
        for crit in criteria:
            print(f"· {crit:<12}: {crit_values[crit]}  (score={crit_scores[crit]:.2f})")


print("\n✅ Evaluación completada; métricas guardadas en MLflow")
