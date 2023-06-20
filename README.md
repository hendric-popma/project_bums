# project_bums


# Hyperopt: Distributed Hyperparameter Optimization

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/de/thumb/d/dc/Hs-kempten-logo.svg/1200px-Hs-kempten-logo.svg.png" />
</p>

[![build](https://github.com/hyperopt/hyperopt/actions/workflows/build.yml/badge.svg)](https://github.com/hyperopt/hyperopt/actions/workflows/build.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/hyperopt/hyperopt/master.svg)](https://results.pre-commit.ci/latest/github/hyperopt/hyperopt/master)
[![PyPI version](https://badge.fury.io/py/hyperopt.svg)](https://badge.fury.io/py/hyperopt)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/hyperopt/badges/version.svg)](https://anaconda.org/conda-forge/hyperopt)

[Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for serial and parallel optimization over awkward
search spaces, which may include real-valued, discrete, and conditional
dimensions.

## Getting started

Install hyperopt from PyPI

```bash
pip install hyperopt
```

to run your first example

```python
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}
```

## Algorithms


## Documentation



## Related Projects


## Examples

## Cite

## Thanks

This project has received support from

