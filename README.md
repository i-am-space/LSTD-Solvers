# Efficient Linear Solvers for LSTD in Reinforcement Learning

This repository explores and benchmarks efficient numerical solvers for **Least-Squares Temporal Difference (LSTD)** learning, a popular method for **policy evaluation** in reinforcement learning (RL). The key objective is to understand the **computational trade-offs** between direct and iterative solvers when solving the core linear system that arises in LSTD.

> **Authors**: Bhavya Ahuja, Mrudani Pimpalkhare, Vishal Rao<br>
> **Institute**: International Institute of Information Technology, Hyderabad<br>
> **Course**: Numerical Algorithms

---

## Motivation

LSTD methods estimate value functions by solving the equation:

$
A w = b, \quad \text{where } A = \sum_t \phi(s_t)(\phi(s_t) - \gamma \phi(s_{t+1}))^T
$

As environments grow in complexity, solving this system becomes increasingly challenging due to:

* High dimensionality (from polynomial or other feature expansions)
* Poor conditioning or low rank (from sparse rewards or chaotic transitions)
* Varying structure across environments (e.g., dense vs. sparse matrices)

Thus, the **choice of solver** is critical for both computational efficiency and solution accuracy.

---

## Goals

* Analyze the **structure of matrix $A$** in LSTD across various RL environments.
* Compare **direct solvers** (LU, QR) with **iterative solvers** (CG, Deflated CG, PyAMG, ILU-CG).
* Understand when and why certain solvers outperform others based on **matrix rank, sparsity, and conditioning**.

---

## Tested Environments

We evaluated solvers across five diverse OpenAI Gym environments:

| Environment | State Space   | Feature Type    | Key Characteristics               |
| ----------- | ------------- | --------------- | --------------------------------- |
| MountainCar | Continuous    | Polynomial      | Smooth transitions, high rank     |
| CartPole    | Continuous    | Polynomial      | Small, chaotic, low-rank          |
| GridWorld   | Discrete Grid | One-hot/tabular | Sparse, grid-like                 |
| Acrobot     | Continuous    | Trig-polynomial | Highly nonlinear, ill-conditioned |
| Pendulum    | Continuous    | Trig-polynomial | Smooth, moderately dense          |

---

## Solvers Implemented

| Solver                          | Type      | Notes                                        |
| ------------------------------- | --------- | -------------------------------------------- |
| **LU Decomposition**            | Direct    | Fast for small systems                       |
| **QR Factorization**            | Direct    | Accurate, stable                             |
| **CG (Conjugate Gradient)**     | Iterative | Only for SPD matrices                        |
| **CG + ILU Preconditioning**    | Iterative | Speed-up for well-structured systems         |
| **Deflated CG**                 | Iterative | Targets low-rank structure                   |
| **PyAMG (Algebraic Multigrid)** | Iterative | Excels in grid-like or smooth feature spaces |

---

## Key Findings

* **LU/QR dominate on small, dense systems** (e.g., CartPole, Acrobot).
* **PyAMG excels in grid-based (FrozenLake) and smooth-feature environments (MountainCar)** due to its ability to leverage structure and sparsity.
* **Deflated CG helps in ill-conditioned, low-rank scenarios**, but adds overhead unless eigenvalue issues dominate.
* **CG with ILU** performs well when the matrix exhibits good locality or continuity.

---

## Solver Selection Strategy

We developed a decision framework to select the ideal solver based on:

* **Matrix Rank**: Low-rank → Use Deflation or direct methods
* **Sparsity**: Sparse + structured → PyAMG
* **Matrix Size**: Small → LU/QR; Large → Iterative methods
* **Environment Smoothness**: Smooth transitions → PyAMG or ILU-CG
* **Chaotic Dynamics**: Poor conditioning → Prefer direct methods

<!-- <p align="center">
  <img src="decision_flowchart.png" width="500"/>
</p> -->

---

## Repository Structure

```
.
├── LSTD_notebook.ipynb       # Main notebook for experiments
├── LSTD_report.pdf           # Full technical report
├── LSTD_ppt.pdf              # Presentation slides
└── README.md                 # This file
```

---

## References

* Bradtke & Barto (1996). "Linear Least-Squares Algorithms for Temporal Difference Learning."
* PyAMG: [https://github.com/pyamg/pyamg](https://github.com/pyamg/pyamg)
* OpenAI Gym environments

---

## Acknowledgements

We thank **Prof. Pawan Kumar** for his guidance in the Numerical Algorithms course and for encouraging a structured experimental approach in analyzing solver performance.

