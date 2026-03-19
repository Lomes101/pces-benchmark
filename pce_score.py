import hashlib, json, numpy as np
from pathlib import Path
import redis, time

def build_graph(seed: str, n_nodes: int, edge_budget: int) -> tuple:
    n_nodes = max(4, n_nodes)  # minimo 4 nodos
    rng = np.random.default_rng(
        int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    nodos = list(range(n_nodes))
    # Construir camino euleriano simple: cadena de nodos
    for k in range(len(nodos)-1):
        i, j = nodos[k], nodos[k+1]
        if adj[i].sum() < edge_budget and adj[j].sum() < edge_budget:
            adj[i][j] = 1
            adj[j][i] = 1
    # Agregar aristas extra aleatorias respetando edge_budget
    pares = [(i,j) for i in range(n_nodes) for j in range(i+1,n_nodes) if adj[i][j]==0]
    rng.shuffle(pares)
    for i,j in pares[:5]:
        if adj[i].sum() < edge_budget and adj[j].sum() < edge_budget:
            adj[i][j] = 1
            adj[j][i] = 1
    restricciones = [
        "max_degree <= " + str(edge_budget),
        "has_eulerian_path",
        "n_nodes=" + str(n_nodes),
        "edges=" + str(int(adj.sum()//2))
    ]
    return adj, restricciones

def score_coherence(pred_adj, true_adj, pred_restr, true_restr) -> float:
    if pred_adj.shape != true_adj.shape:
        matrix_score = 0.0
    else:
        matrix_score = float(np.allclose(pred_adj, true_adj))
    restr_score = float(set(pred_restr) == set(true_restr))
    return round((matrix_score + restr_score) / 2, 4)

def generate_dataset(n_samples=50, output_path="pces_benchmark/dataset.jsonl"):
    Path("pces_benchmark").mkdir(exist_ok=True)
    samples = []
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        seed = hashlib.sha256(("pces_" + str(i)).encode()).hexdigest()[:16]
        n_nodes = int(rng.integers(4, 9))
        edge_budget = int(rng.integers(2, n_nodes))
        adj, restricciones = build_graph(seed, n_nodes, edge_budget)
        samples.append({"id": i, "seed": seed, "n_nodes": n_nodes,
                        "edge_budget": edge_budget,
                        "true_adj": adj.tolist(),
                        "true_restrictions": restricciones})
    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + chr(10))
    print("Dataset: " + str(n_samples) + " samples OK")
    return samples

def baseline_dummy(sample):
    return np.zeros((sample["n_nodes"], sample["n_nodes"]), dtype=np.int8), []

def run_evaluation(dataset_path="pces_benchmark/dataset.jsonl"):
    scores = []
    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            true_adj = np.array(sample["true_adj"], dtype=np.int8)
            pred_adj, pred_restr = baseline_dummy(sample)
            scores.append(score_coherence(pred_adj, true_adj,
                          pred_restr, sample["true_restrictions"]))
    return {"n_samples": len(scores),
            "score_promedio": round(float(np.mean(scores)),4),
            "baseline": "dummy_vacio"}

if __name__ == "__main__":
    print("PCES Benchmark — Coherencia Logica")
    print("Diseno: Taipan | DeepMind 200k")
    print("=" * 40)
    generate_dataset(50)
    resultado = run_evaluation()
    print("Score baseline:", resultado["score_promedio"])
    print("Samples:", resultado["n_samples"])
    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:baseline", json.dumps(resultado), ex=86400*30)
    print("OK")
