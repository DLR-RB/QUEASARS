QUEASARS change log
===================

Current version: 0.3.1

## Unreleased

### Added
### Fixed
### Changed

## 0.3.1

### Added

- Support for python 3.12 ([Issue #46])

## 0.3.0

### Added

- Added tournament selection as an alternative selection method for EVQE ([Issue #25])
- Add the ability to initialize EVQE individuals with more than one circuit layer ([Issue #26])
- Added the ability to use the Critival Value at Risk instead of the Expectation Value ([Issue #32])
- Added JSON serialization classes to enable the serialization of EvolvingAnsatzMinimumEigensolverResults ([Issue #35])
- Added a termination criterion for qiskit_algorithm's SPSA optimizer ([Issue #37])
- Added the ability to use a qiskit PassManager for local quantum circuit transpilation ([Issue #39])

### Fixed

- Fix Pauli strings being in inverse bit order ([Issue #23])
- Fix erroneous interaction of penalty terms in the JSSP Hamiltonian ([Issue #23])
- Fix erroneous normalization of the early start Term in the JSSP Hamiltonian ([Issue #29])

### Changed
- Made termination criteria more consistent ([Issue #31])
- Partially removed the dependence on qiskit_algorithms _DiagonalSampler ([Issue #32])
- Store the amount of expectation value evaluations per generation instead of as one number for the whole optimization ([Issue #35])
- If a circuit is prepended to the individuals for state preparation, store that circuit in the EvolvingAnsatzMinimumEigensolverResult ([Issue #35])
- Update QUEASARS to use the qiskit V2 primitives instead of the qiskit V1 primitives ([Issue #39])
- Require a ConfiguredSampler and ConfiguredEstimator wrapper as input instead of the Sampler and Estimator primitives by themselves ([Issue #39])

## 0.2.0

- Implement a general algorithm structure for Evolving Ansatz VQE algorithms
- Implement the Evolutionary Variational Quantum Eigensolver ([E-VQE](https://arxiv.org/abs/1910.09694))
- Implement Job Shop Scheduling Datastructures and Hamiltonian Encoding, as an example optimization problem

## 0.1.0

- Initial codeless pypi commit

[Issue #46]: https://github.com/DLR-RB/QUEASARS/issues/46
[Issue #39]: https://github.com/DLR-RB/QUEASARS/issues/39
[Issue #37]: https://github.com/DLR-RB/QUEASARS/issues/37
[Issue #35]: https://github.com/DLR-RB/QUEASARS/issues/35
[Issue #32]: https://github.com/DLR-RB/QUEASARS/issues/32
[Issue #31]: https://github.com/DLR-RB/QUEASARS/issues/31
[Issue #29]: https://github.com/DLR-RB/QUEASARS/issues/29
[Issue #26]: https://github.com/DLR-RB/QUEASARS/issues/26
[Issue #25]: https://github.com/DLR-RB/QUEASARS/issues/25
[Issue #23]: https://github.com/DLR-RB/QUEASARS/issues/23