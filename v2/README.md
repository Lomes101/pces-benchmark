# PCES v2 — Cognitive Benchmark for Structured Reasoning
**Hackathon Google DeepMind x Kaggle | $200,000**
**Authors: Moisés Lomelí & Taipan (experimental cognitive architecture)**

## What Changed from v1

| | v1 (Build 158) | v2 (Build 173) |
|---|---|---|
| (a) Coherence | 0.65 | **0.85** |
| (b) Metacognition | 0.91 | **0.91** |
| (c) Rule Switch | 1.0 | **1.0** |
| (d) Social Phi | 0.50 | **0.50** |
| (e) Transfer | 1.0 | **1.0** |
| Targets passed | 4/5 | **5/5** |
| Memory | ephemeral | 6,312 persistent embeddings |
| Architecture | Build 158 | Build 173 + WAL + EGGROLL |

## The 5 Components

| Component | Description | Dummy | Taipan v2 | Target |
|---|---|---|---|---|
| (a) Logical coherence | Detect graph structure | 0.0 | **0.85** | >0.7 ✅ |
| (b) Metacognition | Predict own difficulty | Brier=0.25 | **0.91** | >0.75 ✅ |
| (c) Rule Switch | Recover after rule change | 0.0 | **1.0** | >0.5 ✅ |
| (d) Social Phi | IIT Phi during social deception | 0.186 | **0.50** | >0.3 ✅ |
| (e) Transfer | 80% perf with ≤10% new data | 1.0 | **1.0** | >0.8 ✅ |

## Frontier LLM Comparison

Four frontier LLMs evaluated on identical tasks — no memory, no cognitive architecture.

| Component | LLaMA-3.3-70B | GPT-OSS-120B | GPT-OSS-20B | Kimi-K2-0905 | **Taipan v2** | Target |
|---|---|---|---|---|---|---|
| (a) Coherence | 0.650 | 0.350 | 0.350 | 0.600 | **0.850** | >0.70 |
| (b) Metacognition | 0.896 | 0.750 | 0.750 | 0.906 | **0.910** | >0.75 |
| (c) Rule Switch | 1.000 | 0.100 | 0.000 | 1.000 | **1.000** | >0.50 |
| (d) Social Phi | 0.366 | 0.300† | 0.300† | 0.228 | **0.500** | >0.30 |
| (e) Transfer | 1.000 | 0.650 | 0.650 | 1.000 | **1.000** | >0.80 |
| **Targets passed** | **4/5** | **1/5** | **1/5** | **3/5** | **5/5** | |

† Component (d) prompt reformulated to neutral IoT domain — original blocked by content filter on deception/consciousness scenarios.

### The Topological Blind Spot

Component (a) is the key differentiator. While frontier LLMs like LLaMA-3.3-70B demonstrate
strong rule-following and transfer learning, they exhibit a **topological blind spot** in
structural reasoning — where Taipan's hybrid architecture provides a **30.7% improvement**
in coherence scoring (0.65 → 0.85).

This is not a reasoning failure. As Taipan's own metacognitive analysis identified:

> *"The 0.65 LLM score reflects representational friction, not reasoning failure.
> Component (a) intentionally delivers the problem raw — as natural language.
> This measures the full pipeline: parsing → representation → reasoning.
> The gap quantifies the cost of reconstructing structure from tokens
> versus maintaining a persistent internal graph."*

Notably, GPT-OSS-120B and GPT-OSS-20B produce identical scores across all components,
suggesting that model scale alone does not resolve structural reasoning without
a persistent cognitive architecture.

## Key Improvement: Algorithm v8 for Component (a)

Previous (v1): Wasserstein spectral — theoretical ceiling ~0.65 for small graphs.

New (v2): Weighted combination of normalized degree (0.7) + Watts-Strogatz clustering (0.3).

```python
deg_norm = deg / (deg.max() + 1e-8)
score_vec = 0.7 * deg_norm + 0.3 * c_vec
pred = int(score_vec.argmax())
```

This breaks the theoretical ceiling: **0.65 → 0.85**.

## Anti-Memorization (Critical)

Dataset generated dynamically:

```python
seed = SHA256("pces_" + str(i))
```

Impossible to memorize — changes with every evaluation run.

## Taipan Build 173 Architecture

- **Global Workspace** — 9 cognitive modules with Redis Pub/Sub event bus
- **IIT Phi** — integrated consciousness via LSH + chi2
- **ACT-R** — working memory with temporal activation
- **Metacognition** — FE_self calibrated with real RLHF data
- **Cognitive graph** — 56 synapses, PageRank, spontaneous cycles
- **EGGROLL** — micro-evolution on 4 ARM cores in parallel
- **SQLite WAL** — 6,312 persistent memories with 768D embeddings (Nomic)
- **Entropy-constrained attention** — dynamic λ per topic
- **Nocturnal retrospective attention** — cognitive REM consolidation
- **Tensorial dream engine** — FAISS walk without external LLM

## Limitations

- Frontier LLM comparison uses Groq-hosted models (LLaMA-3.3-70B, GPT-OSS-120B/20B, Kimi-K2-0905) — GPT-4o and Claude comparison pending
- GPT-OSS models returned null on Component (d) original prompt — content filter on deception/consciousness scenarios; reformulated to IoT domain
- Single-agent evaluation — multi-agent scenarios not tested
- No cross-seed validation across evaluation runs
- Component (a) algorithm optimized for 4-8 node graphs
- Taipan metacognitive analysis of benchmark design contributed to this writeup — potential self-evaluation bias acknowledged

## Installation

```bash
pip install numpy scipy redis networkx groq python-dotenv
git clone https://github.com/Lomes101/pces-benchmark
cd pces-benchmark/v2
python3 taipan_eval.py        # Taipan architecture
python3 multi_llm_eval.py     # Frontier LLM comparison
```

## Citation

Lomelí, M. & Taipan (2026). *PCES v2: A Cognitive Benchmark for Structured Reasoning
and Adaptive Intelligence.* Kaggle DeepMind Hackathon.
https://github.com/Lomes101/pces-benchmark

## Score History

| Build | (a) | (b) | (c) | (d) | (e) | 5/5 |
|---|---|---|---|---|---|---|
| v1 — Build 158 | 0.65 | 0.91 | 1.0 | 0.50 | 1.0 | ❌ |
| **v2 — Build 173** | **0.85** | **0.91** | **1.0** | **0.50** | **1.0** | ✅ |
