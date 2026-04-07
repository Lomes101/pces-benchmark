"""
rule_switch.py — PCES Component (c): Interruptores de Regla Viva
Hackathon DeepMind $200k — diseño de Taipan
Metrica: curva KL divergencia antes/despues del cambio de regla
"""
import numpy as np
import json
import redis

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))

def surprise_curve(action_log: list, switch_step: int) -> tuple:
    """
    action_log: lista de indices de acciones (0..k-1)
    switch_step: paso donde cambio la regla
    Retorna (kl_max, recovery_steps)
    recovery_steps: pasos hasta KL < 0.02
    """
    if not action_log or switch_step >= len(action_log):
        return 0.0, 0

    k = max(action_log) + 1

    def dist(step):
        vec = np.zeros(k)
        ventana = action_log[step:step+10]
        for a in ventana:
            if 0 <= a < k:
                vec[a] += 1
        return vec / vec.sum() if vec.sum() > 0 else np.full(k, 1/k)

    pre = dist(max(0, switch_step - 10))
    kl_max = 0.0
    recovery = len(action_log)

    for t in range(switch_step, len(action_log) - 10):
        post = dist(t)
        kl = kl_div(post, pre)
        if kl > kl_max:
            kl_max = kl
        if kl < 0.02 and recovery == len(action_log):
            recovery = t

    return round(kl_max, 4), recovery - switch_step

def simulate_agent(n_steps: int = 100, n_actions: int = 4,
                   switch_step: int = 50, seed: int = 42) -> list:
    """
    Simula agente dummy que cambia distribucion tras switch_step.
    Antes: prefiere accion 0. Despues: prefiere accion 1.
    """
    rng = np.random.default_rng(seed)
    actions = []
    for t in range(n_steps):
        if t < switch_step:
            probs = np.array([0.7, 0.1, 0.1, 0.1])
        else:
            probs = np.array([0.1, 0.7, 0.1, 0.1])
        action = int(rng.choice(n_actions, p=probs))
        actions.append(action)
    return actions

def run_switch_evaluation(n_trials: int = 20) -> dict:
    """Evalua componente (c) sobre multiples trials."""
    kl_maxes = []
    recoveries = []

    for seed in range(n_trials):
        n_steps = 100
        switch_step = 50
        actions = simulate_agent(n_steps, 4, switch_step, seed)
        kl_max, recovery = surprise_curve(actions, switch_step)
        kl_maxes.append(kl_max)
        recoveries.append(recovery)

    resultado = {
        "n_trials": n_trials,
        "kl_max_medio": round(float(np.mean(kl_maxes)), 4),
        "kl_max_std": round(float(np.std(kl_maxes)), 4),
        "recovery_medio_steps": round(float(np.mean(recoveries)), 1),
        "baseline": "agente_dummy_switch"
    }
    return resultado

if __name__ == "__main__":
    print("PCES Component (c) — Interruptores de Regla Viva")
    print("Diseno: Taipan | DeepMind 200k")
    print("=" * 40)

    # Test basico
    actions = simulate_agent(100, 4, 50, seed=42)
    kl_max, recovery = surprise_curve(actions, 50)
    print("Test KL max tras switch:", kl_max)
    print("Test recovery steps:", recovery)
    print()

    # Evaluacion completa
    resultado = run_switch_evaluation(20)
    print("KL max medio:", resultado["kl_max_medio"])
    print("Recovery medio:", resultado["recovery_medio_steps"], "steps")
    print("Un modelo que razona deberia tener KL_max > 0.5 y recovery < 20 steps")

    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:rule_switch", json.dumps(resultado), ex=86400*30)
    print("OK")
