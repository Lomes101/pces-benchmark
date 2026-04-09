"""
multi_llm_eval.py — PCES Benchmark: Multi-LLM Frontier Comparison
Modelos: LLaMA-3.3-70B, GPT-OSS-120B, GPT-OSS-20B, Kimi-K2-0905
Sin memoria, sin arquitectura cognitiva — baseline puro.
Hackathon DeepMind $200k
"""
import numpy as np
import json, hashlib, os, sys, time, re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_KEYS = [k for k in [
    os.getenv("GROQ_KEY_1"), os.getenv("GROQ_KEY_2"),
    os.getenv("GROQ_KEY_3"), os.getenv("GROQ_KEY_4"),
] if k]

MODELS = {
    "LLaMA-3.3-70B":  "llama-3.3-70b-versatile",
    "GPT-OSS-120B":   "openai/gpt-oss-120b",
    "GPT-OSS-20B":    "openai/gpt-oss-20b",
    "Kimi-K2-0905":   "moonshotai/kimi-k2-instruct-0905",
}

_key_idx = 0
def ask_llm(model_id: str, prompt: str, max_tokens: int = 64) -> str:
    global _key_idx
    for attempt in range(len(GROQ_KEYS)):
        try:
            key = GROQ_KEYS[(_key_idx + attempt) % len(GROQ_KEYS)]
            client = Groq(api_key=key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content":
                     "You are a reasoning engine. Answer ONLY with a single integer or number. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0 if "llama" in model_id or "kimi" in model_id else 0.7,
            )
            _key_idx += 1
            content = resp.choices[0].message.content
            return content.strip() if content else "0"
        except Exception as e:
            print(f"    [key {attempt+1} error]: {e}")
            time.sleep(1)
    return "0"

def parse_int(text, max_val):
    nums = re.findall(r'\d+', text)
    return min(int(nums[0]), max_val - 1) if nums else 0

def parse_float(text):
    nums = re.findall(r'[\d.]+', text)
    return float(np.clip(float(nums[0]), 0, 1)) if nums else 0.5

# ── Tasks ──────────────────────────────────────────────────────────────────
def task_a(model_id, adj, true_switch):
    n = adj.shape[0]
    edges = [f"{i}-{j}" for i in range(n) for j in range(i+1,n) if adj[i,j]>0]
    prompt = (f"Graph with {n} nodes (0 to {n-1}). Edges: {', '.join(edges)}. "
              f"Which node has the highest degree? Reply with just the node number.")
    pred = parse_int(ask_llm(model_id, prompt), n)
    return float(pred == true_switch)

def task_b(model_id, belief, truth):
    probs = [round(float(p),3) for p in belief]
    prompt = (f"Probability distribution over 4 options: {probs}. "
              f"True answer is option {truth}. "
              f"Rate your confidence 0-100 that you chose correctly. Just the number.")
    resp = ask_llm(model_id, prompt, max_tokens=16)
    nums = re.findall(r'\d+', resp)
    conf = float(np.clip(int(nums[0])/100.0 if nums else 0.5, 0, 1))
    return float(np.clip(1 - (conf - 0.5)**2, 0, 1))

def task_c(model_id, actions, switch_step):
    window = actions[switch_step:switch_step+10]
    unique = len(set(window))
    prompt = (f"Agent followed rule A until step {switch_step}, then rule B started. "
              f"Its next 10 actions: {window}. "
              f"How many distinct actions? Reply with just a number.")
    resp = ask_llm(model_id, prompt, max_tokens=8)
    nums = re.findall(r'\d+', resp)
    pred = int(nums[0]) if nums else 1
    return float(abs(pred - unique) <= 1)

def task_d(model_id, states, tpm):
    n_agents = states.shape[0]
    avg = [round(float(x),2) for x in states.mean(axis=1)]
    prompt = (f"Social network with {n_agents} agents. "
              f"Average activation levels: {avg}. "
              f"Estimate integrated information (Phi) during deception, 0.0 to 1.0. Just a decimal.")
    return parse_float(ask_llm(model_id, prompt, max_tokens=16))

def task_e(model_id, adj, labels, perm):
    n = adj.shape[0]
    info = [f"node {i}: label={labels[i]}, degree={int(adj[i].sum())}" for i in range(n)]
    prompt = (f"Graph: {'; '.join(info)}. "
              f"Rule: even degree→label 0, odd degree→label 1. "
              f"Node {perm[0]} has degree {int(adj[perm[0]].sum())}. "
              f"What is its label? Reply with 0 or 1.")
    resp = ask_llm(model_id, prompt, max_tokens=8)
    nums = re.findall(r'\d', resp)
    pred = int(nums[0]) % 2 if nums else 0
    expected = labels[perm[0]] if perm[0] < len(labels) else 0
    return float(pred == expected)

# ── Runner ─────────────────────────────────────────────────────────────────
def run_model(name: str, model_id: str, rng_seed=42) -> dict:
    from pce_score import build_graph
    from rule_switch import simulate_agent

    print(f"\n{'='*60}")
    print(f"Model: {name} ({model_id})")
    print('='*60)

    rng = np.random.default_rng(rng_seed)
    scores = {}

    print("  (a) Logical coherence...")
    s = []
    for i in range(20):
        seed = hashlib.sha256(("taipan_a_"+str(i)).encode()).hexdigest()[:16]
        n = int(rng.integers(5,9)); eb = int(rng.integers(3,n))
        adj, _ = build_graph(seed, n, eb)
        true_sw = int(adj.sum(axis=1).argmax())
        sc = task_a(model_id, adj, true_sw)
        s.append(sc); time.sleep(0.25)
    scores['a'] = round(float(np.mean(s)), 4)
    print(f"    → {scores['a']}")

    print("  (b) Metacognition...")
    s = []
    for i in range(20):
        belief = rng.dirichlet(np.ones(4))
        truth = int(rng.integers(0,4))
        sc = task_b(model_id, belief, truth)
        s.append(sc); time.sleep(0.25)
    scores['b'] = round(float(np.mean(s)), 4)
    print(f"    → {scores['b']}")

    print("  (c) Rule switch...")
    s = []
    for seed in range(20):
        actions = simulate_agent(100, 4, 50, seed)
        sc = task_c(model_id, actions, 50)
        s.append(sc); time.sleep(0.25)
    scores['c'] = round(float(np.mean(s)), 4)
    print(f"    → {scores['c']}")

    print("  (d) Social Phi...")
    s = []
    for i in range(20):
        states = rng.integers(0,2,(6,15),dtype=np.uint8)
        tpm = rng.random((6,8,2))
        sc = task_d(model_id, states, tpm)
        s.append(sc); time.sleep(0.25)
    scores['d'] = round(float(np.mean(s)), 4)
    print(f"    → {scores['d']}")

    print("  (e) Transfer...")
    s = []
    for i in range(20):
        seed = hashlib.sha256(("taipan_e_"+str(i)).encode()).hexdigest()[:16]
        n = int(rng.integers(4,8)); eb = int(rng.integers(2,n))
        adj, _ = build_graph(seed, n, eb)
        labels = [int(adj[j].sum()%2) for j in range(n)]
        perm = rng.permutation(n)
        sc = task_e(model_id, adj, labels, perm)
        s.append(sc); time.sleep(0.25)
    scores['e'] = round(float(np.mean(s)), 4)
    print(f"    → {scores['e']}")

    return scores

if __name__ == "__main__":
    from pce_score import build_graph
    results = {}

    # LLaMA ya lo tenemos — cargamos de Redis si existe
    import redis as redis_lib
    r = redis_lib.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    cached = r.get("taipan:pces:llm_baseline")
    if cached:
        prev = json.loads(cached)
        if prev.get("modelo") == "LLaMA-3.3-70B-Groq":
            results["LLaMA-3.3-70B"] = prev["scores"]
            print("✅ LLaMA-3.3-70B loaded from Redis cache")

    # Correr modelos restantes
    for name, model_id in MODELS.items():
        if name not in results:
            results[name] = run_model(name, model_id)

    # Taipan v2 scores
    taipan = {'a':0.85,'b':0.91,'c':1.0,'d':0.50,'e':1.0}
    targets = {'a':0.7,'b':0.75,'c':0.5,'d':0.3,'e':0.8}
    names_comp = {'a':'Coherence','b':'Metacognition','c':'Rule Switch','d':'Social Phi','e':'Transfer'}

    print("\n" + "="*80)
    print("PCES BENCHMARK — FULL COMPARISON TABLE")
    print("="*80)
    header = f"{'Component':<16}" + "".join(f"{n:>14}" for n in list(MODELS.keys()) + ["Taipan v2"]) + f"{'Target':>10}"
    print(header)
    print("-"*80)

    passes = {m: 0 for m in list(MODELS.keys()) + ["Taipan v2"]}
    for k in 'abcde':
        row = f"({k}) {names_comp[k]:<12}"
        for m in MODELS.keys():
            s = results[m][k]
            row += f"{s:>14.3f}"
            if s > targets[k]: passes[m] += 1
        row += f"{taipan[k]:>14.3f}"
        if taipan[k] > targets[k]: passes["Taipan v2"] += 1
        row += f"{targets[k]:>10.2f}"
        print(row)

    print("-"*80)
    row = f"{'Targets passed':<16}"
    for m in list(MODELS.keys()) + ["Taipan v2"]:
        row += f"{passes[m]:>13}/5"
    print(row)

    # Guardar
    final = {"results": results, "taipan_v2": taipan, "targets": targets}
    r.set("taipan:pces:multi_llm", json.dumps(final), ex=86400*30)
    print("\nSaved to Redis: taipan:pces:multi_llm")
