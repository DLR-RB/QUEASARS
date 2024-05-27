QUEASARS change log
===================

Current version: 0.2.0

## Unreleased

### Added

- Added tournament selection as an alternative selection method for EVQE ([Issue #25])
- Add the ability to initialize EVQE individuals with more than one circuit layer ([Issue #26])

### Fixed

- Fix Pauli strings being in inverse bit order ([Issue #23])
- Fix erroneous interaction of penalty terms in the JSSP Hamiltonian ([Issue #23])

## 0.2.0

- Implement a general algorithm structure for Evolving Ansatz VQE algorithms
- Implement the Evolutionary Variational Quantum Eigensolver ([E-VQE](https://arxiv.org/abs/1910.09694))
- Implement Job Shop Scheduling Datastructures and Hamiltonian Encoding, as an example optimization problem

## 0.1.0

- Initial codeless pypi commit

[Issue #26]: https://github.com/DLR-RB/QUEASARS/issues/26
[Issue #25]: https://github.com/DLR-RB/QUEASARS/issues/25
[Issue #23]: https://github.com/DLR-RB/QUEASARS/issues/23