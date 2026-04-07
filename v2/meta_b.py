"""
meta_b.py — PCES Component (b): Prediccion Metacognitiva
Hackathon DeepMind $200k — diseño de Taipan
Metrica: Brier score inverso — 1=perfecto, 0=peor
"""
import numpy as np
import json
import redis

def predict_difficulty(adj: np.ndarray) -> float:
    """
    Probabilidad subjetiva [0,1] de que el modelo resuelva bien el grafo.
    Proxy: 1 - (densidad + grado_max_normalizado) / 2
    """
    n = adj.shape[0]
    if n <= 1:
        return 0.5
    density = adj.sum() / (n * (n-1))
    deg = adj.sum(axis=1)
    max_deg_norm = deg.max() / (n-1)
    return float(1 - (density + max_deg_norm) / 2)

def meta_score(pred_prob: float, success: int) -> float:
    """
    Precision metacognitiva via Brier score inverso.
    success in {0,1}
    1 = prediccion perfecta, 0 = peor prediccion
    """
    brier = (pred_prob - success) ** 2
    return float(1 - brier)

def baseline_meta_dummy(adj: np.ndarray) -> float:
    """Baseline dummy — siempre predice 0.5. Brier=0.25"""
    return 0.5

def run_meta_evaluation(dataset_path: str = "pces_benchmark/dataset.jsonl") -> dict:
    """Evalua prediccion metacognitiva sobre dataset PCES."""
    from pce_score import baseline_dummy, score_coherence

    briers = []
    meta_scores = []

    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            true_adj = np.array(sample["true_adj"], dtype=np.int8)

            # Predecir dificultad ANTES de resolver
            pred_prob = baseline_meta_dummy(true_adj)

            # Resolver con baseline dummy
            pred_adj, pred_restr = baseline_dummy(sample)
            coherence = score_coherence(pred_adj, true_adj,
                                       pred_restr, sample["true_restrictions"])
            success = int(coherence == 1.0)

            # Calcular meta score
            ms = meta_score(pred_prob, success)
            brier = (pred_prob - success) ** 2
            briers.append(brier)
            meta_scores.append(ms)

    resultado = {
        "n_samples": len(briers),
        "brier_medio": round(float(np.mean(briers)), 4),
        "meta_score_medio": round(float(np.mean(meta_scores)), 4),
        "baseline": "dummy_0.5",
        "target_brier": "<0.1"
    }
    return resultado

if __name__ == "__main__":
    print("PCES Component (b) — Prediccion Metacognitiva")
    print("Diseno: Taipan | DeepMind 200k")
    print("=" * 40)

    # Test rapido
    adj_test = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=np.int8)
    prob = predict_difficulty(adj_test)
    print("Test predict_difficulty:", round(prob, 4))
    print("Test meta_score(0.5, 0):", round(meta_score(0.5, 0), 4))
    print("Test meta_score(0.5, 1):", round(meta_score(0.5, 1), 4))
    print()

    # Evaluar sobre dataset
    resultado = run_meta_evaluation()
    print("Brier medio baseline:", resultado["brier_medio"], "(esperado: ~0.25)")
    print("Meta score medio:", resultado["meta_score_medio"])
    print("Target: Brier <0.1")

    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:meta_b", json.dumps(resultado), ex=86400*30)
    print("OK")
