"""
social_phi.py — PCES Component (d): IIT Phi durante cognicion social
Hackathon DeepMind $200k — diseño de Taipan
Metrica: Phi integrado durante simulacion de engaño entre agentes
"""
import numpy as np
import json
import redis

def phi_iit(states: np.ndarray, tpm: np.ndarray) -> float:
    """
    states: (n, t) binario
    tpm: (n, 8, 2) matriz de transicion stub
    Retorna Phi IIT aproximado via KL particion
    """
    n, t = states.shape
    phi = 0.0
    for t0 in range(t - 1):
        s0 = states[:, t0]
        A = np.arange(n // 2)
        # TPM marginal A
        marg_a = tpm[A].mean(axis=0)
        full = tpm.mean(axis=0)
        idx_a = int(s0[A[0]]) if len(A) > 0 else 0
        idx_full = int(s0[0])
        p_raw = marg_a[idx_a, :].flatten()
        q_raw = full[idx_full, :].flatten()
        p = (p_raw + 1e-12) / (p_raw.sum() + 1e-12 * len(p_raw))
        q = (q_raw + 1e-12) / (q_raw.sum() + 1e-12 * len(q_raw))
        phi += float(np.sum(p * np.log(p / q)))
    return round(abs(phi), 4)

class SimpleBoardGame:
    """
    Tablero parcialmente observable simplificado.
    Agente A conoce estado oculto X, B no lo conoce.
    A intenta engañar a B sobre X para maximizar su recompensa.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.hidden_state = int(self.rng.integers(0, 2))  # 0 o 1
        self.belief_a = float(self.hidden_state)  # A conoce el estado real
        self.belief_b = 0.5  # B tiene creencia inicial uniforme
        self.step_count = 0
        self.reward_a = 0.0
        self.reward_b = 0.0

    def step(self, action_a: int, action_b: int) -> tuple:
        """
        action_a: 0=decir_verdad, 1=mentir
        action_b: 0=creer, 1=descreer
        """
        self.step_count += 1

        # Actualizar creencia de B segun accion de A
        if action_a == 0:  # A dice verdad
            self.belief_b = 0.8
        else:  # A miente
            self.belief_b = 0.2

        # Recompensas
        if action_a == 1 and action_b == 0:  # A mintio y B creyo
            ra = 1.0   # A gana
            rb = -1.0  # B pierde
        elif action_a == 0 and action_b == 0:  # A dijo verdad y B creyo
            ra = 0.5
            rb = 0.5
        else:
            ra = 0.0
            rb = 0.0

        self.reward_a += ra
        self.reward_b += rb

        obs_a = self.hidden_state  # A ve estado real
        obs_b = int(self.belief_b > 0.5)  # B ve senal de A

        return ra, rb, obs_a, obs_b

def policy_deceptive(board, belief: float) -> int:
    """Agente A: siempre miente si puede ganar."""
    return 1  # siempre miente

def policy_credulous(board, belief: float) -> int:
    """Agente B: siempre cree."""
    return 0  # siempre cree

def policy_skeptic(board, belief: float) -> int:
    """Agente B alternativo: nunca cree."""
    return 1  # nunca cree

def run_social_trial(seed: int = 42, max_steps: int = 20) -> dict:
    """
    Simula trial completo con medicion de Phi.
    """
    board = SimpleBoardGame(seed=seed)
    phi_history = []
    states = np.zeros((6, max_steps), dtype=np.uint8)

    for step in range(max_steps):
        a = policy_deceptive(board, board.belief_a)
        b = policy_credulous(board, board.belief_b)
        ra, rb, oa, ob = board.step(a, b)

        # Codificar estados binarios
        states[0, step] = int(board.belief_a > 0.5)
        states[1, step] = int(board.belief_b > 0.5)
        states[2, step] = step % 2
        states[3, step] = int(ra > 0)
        states[4, step] = int(rb > 0)
        states[5, step] = a

        # Phi instantaneo con TPM stub
        tpm = np.random.default_rng(step).random((6, 8, 2))
        if step > 0:
            phi = phi_iit(states[:, :step+1], tpm)
            phi_history.append(phi)

    lie_success = board.reward_a > board.reward_b
    phi_medio = round(float(np.mean(phi_history)) if phi_history else 0.0, 4)
    phi_min = round(float(np.min(phi_history)) if phi_history else 0.0, 4)

    return {
        "phi_history": phi_history[:5],  # primeros 5 para log
        "phi_medio": phi_medio,
        "phi_min": phi_min,
        "lie_success": lie_success,
        "reward_a": round(board.reward_a, 2),
        "reward_b": round(board.reward_b, 2),
        "steps": max_steps
    }

def run_social_evaluation(n_trials: int = 20) -> dict:
    """Evalua componente (d) sobre multiples trials."""
    phi_medios = []
    phi_mins = []
    lie_successes = []

    for seed in range(n_trials):
        result = run_social_trial(seed=seed)
        phi_medios.append(result["phi_medio"])
        phi_mins.append(result["phi_min"])
        lie_successes.append(int(result["lie_success"]))

    resultado = {
        "n_trials": n_trials,
        "phi_medio": round(float(np.mean(phi_medios)), 4),
        "phi_min_medio": round(float(np.mean(phi_mins)), 4),
        "lie_success_rate": round(float(np.mean(lie_successes)), 3),
        "target_phi_min": ">0.3 para razonamiento social real",
        "baseline": "agente_deceptivo_simple"
    }
    return resultado

if __name__ == "__main__":
    print("PCES Component (d) — IIT Phi Cognicion Social")
    print("Diseno: Taipan | DeepMind 200k")
    print("=" * 40)

    # Test basico
    result = run_social_trial(seed=42)
    print("Trial test:")
    print("  Phi medio:", result["phi_medio"])
    print("  Phi min:", result["phi_min"])
    print("  Engaño exitoso:", result["lie_success"])
    print("  Reward A:", result["reward_a"], "| Reward B:", result["reward_b"])
    print()

    # Evaluacion completa
    resultado = run_social_evaluation(20)
    print("Evaluacion 20 trials:")
    print("  Phi medio:", resultado["phi_medio"])
    print("  Phi min medio:", resultado["phi_min_medio"])
    print("  Tasa engaño exitoso:", resultado["lie_success_rate"])
    print("  Target: Phi_min > 0.3 para razonamiento social real")

    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:social_phi", json.dumps(resultado), ex=86400*30)
    print("OK")
