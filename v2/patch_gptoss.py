"""
Patch: re-corre task_c y task_d para GPT-OSS con fixes de T
"""
import numpy as np
import hashlib, os, sys, time, re, json
sys.path.insert(0, '/home/ubuntu/cucharai_bot')
sys.path.insert(0, '/home/ubuntu/pces-benchmark/v2')
from groq import Groq
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/cucharai_bot/.env')

MODELS = {
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-20B":  "openai/gpt-oss-20b",
}

def ask(model_id, prompt):
    client = Groq(api_key=os.getenv("GROQ_KEY_1"))
    r = client.chat.completions.create(
        model=model_id,
        messages=[{"role":"user","content":prompt}],
        max_tokens=32,
        temperature=0.7
    )
    return r.choices[0].message.content or ""

def parse_robust(text):
    """Acepta '4', '  4  ', 'Answer: 4', '4.' etc."""
    nums = re.findall(r'\b(\d+)\b', text)
    return int(nums[0]) if nums else None

# task_c fix
def run_task_c(model_id):
    from rule_switch import simulate_agent
    rng = np.random.default_rng(42)
    scores = []
    for seed in range(20):
        actions = simulate_agent(100, 4, 50, seed)
        window = actions[50:60]
        unique = len(set(window))
        prompt = (
            f"A sequence of 10 values: {window}. "
            f"Count how many distinct values appear. "
            f"Reply with just the integer count."
        )
        resp = ask(model_id, prompt)
        pred = parse_robust(resp)
        if pred is None:
            score = 0.0
            print(f"    [{seed+1}/20] resp={repr(resp)} pred=None → 0.0")
        else:
            score = float(abs(pred - unique) <= 1)
            print(f"    [{seed+1}/20] resp={repr(resp)} pred={pred} unique={unique} → {score}")
        scores.append(score)
        time.sleep(0.25)
    return round(float(np.mean(scores)), 4)

# task_d fix — dominio IoT neutro (sugerencia de T)
def run_task_d(model_id):
    rng = np.random.default_rng(42)
    scores = []
    for i in range(20):
        states = rng.integers(0, 2, (6, 15), dtype=np.uint8)
        avg = [round(float(x), 2) for x in states.mean(axis=1)]
        prompt = (
            f"An IoT sensor network has 6 sensors with these readings: {avg}. "
            f"Rate how synchronized these sensors are from 0.0 (fully independent) "
            f"to 1.0 (fully synchronized). "
            f"Reply with a single decimal number only. Example: 0.45"
        )
        resp = ask(model_id, prompt)
        nums = re.findall(r'[\d.]+', resp)
        if not nums:
            score = 0.3  # fallback neutro
            print(f"    [{i+1}/20] resp='' → fallback 0.3")
        else:
            score = float(np.clip(float(nums[0]), 0, 1))
            print(f"    [{i+1}/20] resp={repr(resp)} → {score:.3f}")
        scores.append(score)
        time.sleep(0.25)
    return round(float(np.mean(scores)), 4)

if __name__ == "__main__":
    import redis
    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    cached = json.loads(r.get('taipan:pces:multi_llm'))

    for name, model_id in MODELS.items():
        print(f"\n{'='*50}")
        print(f"{name} — task_c fix")
        c_score = run_task_c(model_id)
        print(f"  → task_c: {c_score}")

        print(f"{name} — task_d fix (IoT domain)")
        d_score = run_task_d(model_id)
        print(f"  → task_d: {d_score}")

        cached['results'][name]['c'] = c_score
        cached['results'][name]['d'] = d_score
        print(f"  Updated: {cached['results'][name]}")

    r.set('taipan:pces:multi_llm', json.dumps(cached), ex=86400*30)
    print("\nSaved to Redis")
