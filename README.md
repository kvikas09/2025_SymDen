# Symbolic Density Estimators for Unnormalized Distributions

### Learning the Hamiltonian $H(\mathbf{x})$ of Boltzmann Distributions from Samples

This repository contains the implementation of our framework for estimating **compact symbolic expressions of unnormalized probability distributions directly from observed samples**.

We combine **deep generative models** with **symbolic regression (SR)** and incorporate domain-specific inductive biases, such as interaction ranges and predefined primitive functions, to make symbolic estimation tractable for high-dimensional distributions.

For a Boltzmann distribution,

$$
p(\mathbf{x}) = \frac{1}{Z}\exp[-H(\mathbf{x})],
$$

the goal is to estimate the underlying **Hamiltonian $H(\mathbf{x})$** using only samples from the distribution:

$$
\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N \sim p(\mathbf{x}).
$$

The learned symbolic Hamiltonian provides an interpretable representation of the underlying distribution and can be used to analyze the interactions governing the system.

---

## Method Overview

The proposed framework consists of two main components:

1. **Deep Generative Model**
   - Learns a flexible representation of the underlying probability density from samples.
   - We consider both:
     - Likelihood-based models, such as **Normalizing Flows**
     - Score-based generative models

2. **Symbolic Regression**
   - Uses the learned density/score representation to estimate a compact symbolic expression.
   - Incorporates domain-specific prior knowledge such as:
     - Interaction range
     - Locality of interactions
     - Predefined primitive-function libraries
     - Factorization into smaller local components

This combination allows symbolic expressions to be estimated even for high-dimensional distributions where directly searching over the complete expression space becomes computationally expensive.

---

## Input

The framework assumes that only samples from the target distribution are available:

$$
\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N \sim p(\mathbf{x}).
$$

No analytical expression of the target Hamiltonian is provided during symbolic estimation.

### Goal

Given samples, estimate a compact symbolic approximation:

$$
\hat{H}(\mathbf{x}) \approx H(\mathbf{x}),
$$

such that the corresponding Boltzmann distribution

$$
\hat{p}(\mathbf{x}) \propto \exp[-\hat{H}(\mathbf{x})]
$$

captures the structure of the original distribution.

---

## Datasets

The framework is evaluated on a range of distributions, from low-dimensional toy problems to high-dimensional models from computational physics.

### 1. Two-Dimensional Toy Distributions

#### Independent Multivariate Gaussian
- 2-dimensional Gaussian distribution
- Independent components
- Target symbolic expression contains **5/4 terms**

#### Multivariate Gaussian with Full Covariance
- 2-dimensional Gaussian distribution
- Full covariance matrix
- Target symbolic expression contains **6/5 terms**

#### Double-Well Distribution
- 2-dimensional multimodal distribution
- Samples generated using **Hamiltonian Monte Carlo (HMC)**
- Target symbolic expression contains **4 terms**

---

### 2. Many-Well Distribution

A high-dimensional multimodal distribution designed to test the scalability of symbolic estimation.

- Dimensions: **$d = 4, 8, 16, 32, 64$**
- Multiple interacting wells
- Used to evaluate the ability of the framework to recover symbolic structure as dimensionality increases

---

### 3. XY Model

A classical statistical-mechanics model with local interactions on a two-dimensional lattice.

- Lattice dimension: **$8 \times 8$**
- Total dimension: **$d = 64$**
- Used to evaluate recovery of local interaction structure from samples

---

### 4. Scalar $\phi^4$ Theory

A lattice field-theory model with nonlinear local interactions.

- **$8 \times 8$ lattice:** $d = 64$
- **$32 \times 32$ lattice:** $d = 1024$
- Used to evaluate symbolic estimation in high-dimensional systems

The framework can also be applied to the **renormalization problem**, where symbolic approximations of effective Hamiltonians can be estimated at different spatial scales directly from samples.

---

## Key Features

- 📌 **Sample-based symbolic estimation** of unnormalized distributions
- 🧠 Combines **deep generative models + symbolic regression**
- 🔬 Supports **likelihood-based and score-based generative models**
- 🧩 Incorporates **domain-specific inductive biases**
- 📐 Exploits **local interaction structure and factorization**
- 📊 Applicable to both **toy distributions and computational-physics models**
- 🔍 Produces **compact and interpretable symbolic Hamiltonians**
- 📈 Scales to high-dimensional systems, including **$d=1024$**

---

## Applications

The framework is particularly relevant for:

- Symbolic modeling of probability distributions
- Interpretable generative modeling
- Statistical physics
- Computational physics
- Learning effective Hamiltonians
- Hamiltonian estimation from data
- Renormalization and coarse-graining
- Scientific machine learning

---
<!--
## Citation

If you find this work useful, please consider citing:

```bibtex
@article{kanaujia2026symbolic,
  title={Symbolic Density Estimators for Unnormalized Distributions},
  author={Kanaujia, Vikas and Arora, Vipul},
  journal={Transactions on Machine Learning Research},
  year={2026}
}-->
