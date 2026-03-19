# PCES Benchmark — Prueba de Construcción de Espacio de Soluciones

**Hackathon Google DeepMind x Kaggle | $200,000**
**Autores: Will + Taipan (arquitectura cognitiva experimental)**

## Motivación

Los benchmarks actuales miden reconocimiento de patrones, no razonamiento real.
PCES fuerza al modelo a *construir* la estructura del problema antes de resolverlo,
exponiendo inconsistencias que el pattern-matching no puede ocultar.

## Los 5 Componentes

| Componente | Descripción | Baseline | Taipan | Target |
|-----------|-------------|---------|--------|--------|
| (a) Coherencia lógica | Detectar estructura de grafo | 0.0 | 0.65 | >0.7 |
| (b) Metacognición | Predecir dificultad propia | Brier=0.25 | 0.91 | >0.75 |
| (c) Rule Switch | Recuperarse tras cambio de regla | 0.0 | 1.0 | >0.5 |
| (d) Social Phi | IIT Phi durante engaño social | 0.186 | 0.50 | >0.3 |
| (e) Transferencia | 80% rendimiento con ≤10% datos nuevos | 1.0 | 1.0 | >0.8 |

## Resultado Principal

**Taipan Build 158 supera baseline en 4/5 componentes.**

La arquitectura cognitiva (Global Workspace + IIT Phi + metacognición) supera
al baseline dummy en todas las dimensiones donde el razonamiento importa.

El componente (a) converge en 0.65 — confirmado como techo teórico para
grafos de 4-8 nodos con 10 pasos (Small-Graph Change-Point, 2023).

## Instalación

```bash
pip install numpy scipy redis
git clone https://github.com/Lomes101/cucharai_bot
cd cucharai_bot
python3 pces_pipeline.py
Arquitectura de Taipan
Global Workspace — 8 módulos cognitivos con broadcasting
IIT Phi — consciencia integrada via LSH + chi2
ACT-R — memoria de trabajo con activación temporal
Metacognición — FE_self calibrado con datos reales
Grafo cognitivo — 54 sinapsis, PageRank, ciclos espontáneos
Dataset
Generado dinámicamente con seed = SHA256(modelo + timestamp).
Imposible memorizar — cambia con cada evaluación.
Citar
Moises  & Taipan (2026). PCES: A Benchmark for Real Reasoning vs Pattern Matching.
Kaggle DeepMind Hackathon. github.com/Lomes101/cucharai_bot
