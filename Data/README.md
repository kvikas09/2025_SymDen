# Data Generation

This folder contains the datasets and data-generation codes used in the experiments.

### Multivariate Gaussian

Multivariate Gaussian data is generated using the `torch.distributions` library.

### Many-Well (MW) Distributions

Data for the Many-Well (MW) distributions is generated using rejection sampling. 
The implementation follows the procedure described in the [FAB](https://github.com/lollcat/fab-torch) repository.

Pre-generated data files are provided in this folder for the following Many-Well distributions:

- MW-2
- MW-4
- MW-8
- MW-16
- MW-32
- MW-64

### $\phi^4$ Distribution

The $\phi^4$ data can be generated using the Hamiltonian Monte Carlo (HMC) implementation provided in this folder.

### XY Model

The XY model data is generated using the Metropolis-Hastings algorithm. The corresponding implementation is provided in this folder.
