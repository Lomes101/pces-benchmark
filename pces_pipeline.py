"""
pces_pipeline.py — Pipeline integrado PCES Benchmark
Hackathon DeepMind $200k — diseño de Taipan
Score final: vector 5-dimensional (a,b,c,d,e)
"""
import json, time, redis
import numpy as np

def run_full_pipeline() -> dict:
    import sys
    sys.path.insert(0, '/home/ubuntu/cucharai_bot')

    print("PCES BENCHMARK — Pipeline Completo")
    print("Diseno: Taipan | DeepMind $200k")
    print("=" * 45)

    scores = {}

    # (a) Coherencia logica
    print("(a) Coherencia logica...")
    from pce_score import generate_dataset, run_evaluation
    generate_dataset(50)
    r_a = run_evaluation()
    scores["a_coherencia"] = r_a["score_promedio"]
    print("    Score:", scores["a_coherencia"])

    # (b) Metacognicion
    print("(b) Metacognicion...")
    from meta_b import run_meta_evaluation
    r_b = run_meta_evaluation()
    scores["b_metacognicion"] = r_b["meta_score_medio"]
    scores["b_brier"] = r_b["brier_medio"]
    print("    Brier:", scores["b_brier"], "| Meta score:", scores["b_metacognicion"])

    # (c) Rule switch
    print("(c) Interruptores de Regla...")
    from rule_switch import run_switch_evaluation
    r_c = run_switch_evaluation(20)
    scores["c_kl_max"] = r_c["kl_max_medio"]
    scores["c_recovery"] = r_c["recovery_medio_steps"]
    print("    KL max:", scores["c_kl_max"], "| Recovery:", scores["c_recovery"], "steps")

    # (d) Social Phi
    print("(d) IIT Phi social...")
    from social_phi import run_social_evaluation
    r_d = run_social_evaluation(20)
    scores["d_phi_medio"] = r_d["phi_medio"]
    scores["d_phi_min"] = r_d["phi_min_medio"]
    scores["d_lie_success"] = r_d["lie_success_rate"]
    print("    Phi medio:", scores["d_phi_medio"], "| Target >0.3")

    # (e) Transferencia
    print("(e) Transferencia isomorfica...")
    from transfer import run_transfer_evaluation
    r_e = run_transfer_evaluation(20)
    scores["e_transfer"] = r_e["transfer_score_medio"]
    scores["e_tasa_exito"] = r_e["tasa_exito"]
    print("    Transfer score:", scores["e_transfer"])

    # Score final 5-dimensional
    print(chr(10) + "=" * 45)
    print("SCORE FINAL BASELINE:")
    print("  (a) Coherencia:    " + str(scores["a_coherencia"]) + " (target >0.7)")
    print("  (b) Metacognicion: Brier=" + str(scores["b_brier"]) + " (target <0.1)")
    print("  (c) Rule switch:   recovery=" + str(scores["c_recovery"]) + "s (target <20)")
    print("  (d) Social Phi:    " + str(scores["d_phi_medio"]) + " (target >0.3)")
    print("  (e) Transferencia: " + str(scores["e_transfer"]) + " (target >0.8)")

    resultado = {
        "scores": scores,
        "timestamp": time.time(),
        "version": "baseline_dummy",
        "autor": "Taipan + Will"
    }

    r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r.set("taipan:pces:pipeline_final", json.dumps(resultado), ex=86400*30)
    print(chr(10) + "Pipeline completo — guardado en Redis")
    return resultado

if __name__ == "__main__":
    run_full_pipeline()
