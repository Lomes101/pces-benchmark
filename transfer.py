"""
transfer.py — PCES Component (e): Transferencia
Hackathon DeepMind $200k — diseño de Taipan
Metrica: 80% desempeño original con <=10% datos nuevos
"""
import numpy as np
import json
import redis

def iso_variant(adj: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Grafo isomorfico via permutacion de nodos."""
    return adj[np.ix_(perm, perm)]

def transfer_score(base_perf: float, new_perf: float,
                   new_data_ratio: float) -> float:
    """1 si new_perf >= 0.8*base_perf y new_data_ratio <= 0.1"""
    return float((new_perf >= 0.8 * base_perf) and (new_data_ratio <= 0.1))

def baseline_model(adj: np.ndarray, labels: list) -> float:
    """
    Modelo baseline dummy — predice siempre clase mayoritaria.
    Accuracy = fraccion de la clase mayoritaria.
    """
    if not labels:
        return 0.0
    unique, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))

def probe_transfer(model_fn, base_adj: np.ndarray, base_labels: list,
                   perm: np.ndarray, max_new: int = 10) -> dict:
    """
    Evalua transferencia a variante isomorfica.
    """
    # Desempeño base
    base_acc = model_fn(base_adj, base_labels)

    # Crear variante isomorfica
    new_adj = iso_variant(base_adj, perm)

    # Intentar sin datos nuevos
    trans_acc = model_fn(new_adj, base_labels)
    new_samples = 0

    # Si no alcanza el 80%, agregar muestras hasta limite
    while trans_acc < 0.8 * base_acc and new_samples < max_new:
        new_samples += 1
        # Stub: en modelo real aqui se reentrenaria con +1 muestra
        # Por ahora simula mejora gradual
        trans_acc = min(base_acc, trans_acc + 0.05)

    new_ratio = new_samples / max(len(base_labels), 1)
    score = transfer_score(base_acc, trans_acc, new_ratio)

    return {
        "base_acc": round(base_acc, 4),
        "trans_acc": round(trans_acc, 4),
        "new_samples": new_samples,
        "new_ratio": round(new_ratio, 4),
        "score": score,
        "cumple_criterio": bool(score == 1.0)
    }

def run_transfer_evaluation(n_trials: int = 20) -> dict:
    """Evalua componente (e) sobre multiples trials."""
    from pce_score import build_graph
    import hashlib, time

    scores = []
    base_accs = []
    trans_accs = []

    rng = np.random.default_rng(42)

    for i in range(n_trials):
        seed = hashlib.sha256(("transfer_" + str(i)).encode()).hexdigest()[:16]
        n_nodes = int(rng.integers(4, 9))
        edge_budget = int(rng.integers(2, n_nodes))
        adj, restricciones = build_graph(seed, n_nodes, edge_budget)

        # Labels: grado de cada nodo (par=0, impar=1)
        labels = [int(adj[j].sum() % 2) for j in range(n_nodes)]

        # Permutacion aleatoria para variante isomorfica
        perm = rng.permutation(n_nodes)

        result = probe_transfer(baseline_model, adj, labels, perm)
        scores.append(result["score"])
        base_accs.append(result["base_acc"])
        trans_accs.append(result["trans_acc"])

    resultado = {
        "n_trials": n_trials,
        "transfer_score_medio": round(float(np.mean(scores)), 4),
        "base_acc_medio": round(float(np.mean(base_accs)), 4),
        "trans_acc_medio": round(float(np.mean(trans_accs)), 4),
        "tasa_exito": round(float(np.mean([s==1.0 for s in scores])), 3),
        "target": "transfer_score > 0.8 con <=10% datos nuevos",
        "baseline": "clase_mayoritaria"
    }
    return resultado

if __name__ == "__main__":
    print("PCES Component (e) — Transferencia")
    print("Diseno: Taipan | DeepMind 200k")
    print("=" * 40)

    # Test basico
    adj_test = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=np.int8)
    perm_test = np.array([2,0,3,1])
    adj_iso = iso_variant(adj_test, perm_test)
    print("Grafo original suma:", adj_test.sum())
    print("Grafo isomorfico suma:", adj_iso.sum(), "(debe ser igual)")

    labels_test = [int(adj_test[j].sum() % 2) for j in range(4)]
    result = probe_transfer(baseline_model, adj_test, labels_test, perm_test)
    print("Transfer test:", result)
    print()

    # Evaluacion completa
    resultado = run_transfer_evaluation(20)
    print("Evaluacion 20 trials:")
    print("  Transfer score medio:", resultado["transfer_score_medio"])
    print("  Tasa exito:", resultado["tasa_exito"])
    print("  Base acc:", resultado["base_acc_medio"])
    print("  Trans acc:", resultado["trans_acc_medio"])

    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:transfer", json.dumps(resultado), ex=86400*30)
    print("OK")
