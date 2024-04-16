# QUEASARS - Quantum Evolving Ansatz Variational Solver

QUEASARS is an open-source, qiskit-based, python package implementing quantum variational eigensolvers which use evolutionary algorithms to find a good ansatz during the optimization process, like [E-VQE](https://arxiv.org/abs/1910.09694), [MoG-VQE](https://arxiv.org/abs/2007.04424) or [QNEAT](https://arxiv.org/abs/2304.06981).
Currently only EVQE is implemented.

QUEASARS is developed as part of a research project of the Quantum Space Operations Center ([QSOC](https://qsoc.space)) at the German Space Operations Center ([GSOC](https://www.dlr.de/en/research-and-transfer/projects-and-missions/iss/the-german-space-operations-center)).

Table of contents
-----------------

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [Copyright and licenses](#copyright-and-licenses)
  - [Copyright](#copyright)
  - [License](#license)

Installation
------------

### Using Pip

QUEASARS requires a python3 environment with python >= 3.9 and can be installed using the following pip command:

      pip install queasars

### From Source

QUEASARS' development dependencies are managed using [poetry](https://github.com/python-poetry/poetry).
To install QUEASARS from source follow these instructions:

1. Clone the QUEASARS repository.
2. Install Python 3.11
3. Install poetry ([installation guide](https://python-poetry.org/docs/#installing-with-pipx)).
4. Run `poetry install` from within QUEASARS' project directory to install its dependencies.


Usage
-----

Documentation
-------------

A more detailed documentation is available at [https://dlr-rb.github.io/QUEASARS/](https://dlr-rb.github.io/QUEASARS/).

Contributing
------------

Contributions to this project are welcome. You may open issues, fix or expand documentation, provide new functionality or create more and better tests. If you have a minor contribution you can open a pull request right away. For any major contribution please open an issue first or discuss with the repository maintainer. Please also note that you need to fill out and sign a [contributor license agreement](DLR%20Individual%20Contributor%20License%20Agreement.pdf)

Maintainers
-----------

The current Maintainers of QUEASARS are [Sven Prüfer (@svenpruefer)](https://github.com/svenpruefer) and [Daniel Leidreiter (@dleidreiter)](https://github.com/dleidreiter).
QUEASARS is currently being developed within the context of Daniel Leidreiter's master thesis.

Copyright and license
---------------------

### Copyright

Quantum Evolving Ansatz Variational Solver (QUEASARS)

Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

This product was developed at DLR - GSOC (German Space Operations Center at the German Aerospace Center DLR, https://www.dlr.de/).

### License

QUEASARS is licensed under the [Apache License, Version 2.0](LICENSE.txt).