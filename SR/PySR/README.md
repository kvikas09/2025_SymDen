# Symbolic Regression of Local Lattice Actions

This directory contains notebooks for extracting and reconstructing **local lattice-field-theory expressions using symbolic regression**.

The analysis uses local field configurations together with quantities obtained from generative models or score functions. A local patch of the lattice field is used as the input to **PySR**, with the goal of identifying a compact analytic expression for the corresponding local contribution.

The same analysis framework is used for the scalar \(\phi^4\) theory, a multi-well scalar model, and the XY model. The XY-model notebooks are not included in this subsection, but follow the same methodology as the \(\phi^4\) analysis.

## Contents

| Notebook              | Description                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `Phi4_NF.ipynb`       | Symbolic regression using data generated from a normalizing-flow (NF) model for lattice \(\phi^4\) theory |
| `Phi4_score.ipynb`    | Symbolic regression of the score/local structure for lattice \(\phi^4\) theory                            |
| `many_wells_NF.ipynb` | Symbolic regression using normalizing-flow data for a many-well scalar model                              |
| `many_wells_SM.ipynb` | Symbolic regression using score-model data for the same many-well model                                   |

The corresponding **XY-model** analysis uses the same local-patch and symbolic-regression strategy as the \(\phi^4\) analysis.

---

## Method

The basic procedure is:

1. Load lattice configurations and the corresponding target quantity.
2. Extract a local lattice patch around each site.
3. Use the local field variables as features for symbolic regression.
4. Fit a symbolic expression using **PySR**.
5. Convert the resulting expression to a SymPy representation.
6. Evaluate the reconstructed expression and compare it with the target using quantities such as the **mean-squared error (MSE)** and \(R^2\).

The local nature of the input is important: rather than fitting an expression to the complete lattice configuration, the regression is performed using a small neighborhood around each lattice site.

---

## Local patches

For the \(\phi^4\) analyses, the notebooks construct local lattice features from neighboring field values with periodic (`wrap`) boundary conditions.

For example, `Phi4_score.ipynb` uses the center site and its nearest neighbors,

$$
(\phi_x,\phi_{x+\hat x},\phi_{x-\hat x},
  \phi_{x+\hat y},\phi_{x-\hat y}),
$$

as the local input to the symbolic regression.

The normalizing-flow \(\phi^4\) notebook uses a corresponding local construction for its regression problem.

For the many-well analyses, the regression is performed on pairs of neighboring field values. The configurations are sampled at selected lattice positions so that the local expression can be reconstructed from the reduced set of variables.

---

## Symbolic regression

The notebooks use [`PySR`](https://astroautomata.com/PySR/) to search over analytic expressions.

The searches use combinations of

* addition,
* subtraction,
* multiplication,
* powers such as \(x^2\) and \(x^3\),

with bounded expression complexity.

The resulting symbolic expressions are converted to SymPy expressions for further manipulation. In the \(\phi^4\) score analysis, the learned score expression is also integrated symbolically to reconstruct a corresponding potential-like expression.

---

## \(\phi^4\) theory

### `Phi4_NF.ipynb`

This notebook analyzes data associated with a normalizing-flow model trained for lattice \(\phi^4\) theory.

The notebook:

* loads the NF-generated dataset,
* extracts local lattice patches,
* constructs the regression targets,
* performs symbolic regression with PySR,
* obtains the resulting symbolic expression,
* evaluates the reconstruction,
* computes MSE and \(R^2\).

The regression is performed configuration-wise, with the local predictions combined to reproduce the corresponding global target.

### `Phi4_score.ipynb`

This notebook performs a related analysis using score data for an \(8\times8\) lattice.

The local input consists of the center field and its four nearest neighbors. PySR is then used to identify an analytic representation of the score.

The learned score can subsequently be integrated symbolically. This provides a route from a learned local score,

$$
s(\phi_x,\ldots),
$$

to a corresponding local potential/action contribution,

$$
V(\phi_x,\ldots).
$$

The notebook also evaluates the reconstructed expression against the original data using MSE and \(R^2\).

---

## Many-well model

The many-well model provides a more nontrivial test of whether a local analytic structure can be recovered from generated data.

### `many_wells_NF.ipynb`

This notebook uses normalizing-flow data for a \(64\)-site many-well system.

A local pair of neighboring field values is used as the symbolic-regression input. The PySR loss is constructed so that the local predictions are combined over the lattice before comparison with the corresponding global target.

### `many_wells_SM.ipynb`

This notebook applies the same general idea using score-model data.

The local symbolic expression is learned from neighboring field values and is subsequently integrated to obtain a potential-like expression. This allows the learned local structure to be compared with the underlying many-well model.

---

## XY model

The same symbolic-regression framework is also applied to the **XY model**.

The XY-model notebooks are not included in this subsection, but they follow the same general pipeline:

$$
\text{lattice configurations}
\;\longrightarrow\;
\text{local patches}
\;\longrightarrow\;
\text{symbolic regression}
\;\longrightarrow\;
\text{local analytic expression}.
$$

Thus, the \(\phi^4\), many-well, and XY-model analyses provide different test cases for recovering local lattice-field-theory structure from generated or learned data.

---

## Data

The notebooks expect the corresponding datasets used in the larger repository. The input datasets are generally stored as pickled objects containing lattice configurations and associated target quantities.

The exact dataset paths appearing in the notebooks refer to files in the parent project, for example:

```text
RG/test_data_phi4_32x32_NF_trained.pkl
phi4_8x8_lamda_4_mass_1.pkl
many_well_dataset_64_with_neg_logp.pkl
many_well_data_64_dim_with_neg_score_through_score_model_100000.pkl
```

These datasets are not necessarily part of this subsection and should be obtained from the corresponding data-generation sections of the main repository.

---

## Requirements

The notebooks use Python packages including:

```text
numpy
torch
pysr
sympy
scikit-learn
```

PySR requires a working Julia installation and its associated Python interface.

The notebooks were developed as exploratory research notebooks, so some paths and dataset locations refer to the directory structure of the larger project.

---

## Purpose

The main purpose of this subsection is to investigate whether **local analytic lattice structures can be recovered from data produced by generative or score-based models**.

In particular, the symbolic-regression approach provides a way to move from numerical data,

$$
\{\phi,\; \text{learned quantity}\},
$$

to an interpretable analytic representation,

$$
\text{local field variables}
\quad\longrightarrow\quad
\text{symbolic expression}.
$$

The \(\phi^4\), many-well, and XY-model examples test this idea across different lattice-field-theory settings.

