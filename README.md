<p align="center">
    <img src="docs/social-banner-bg-rounded.png" height="128" width="384"/>
    <br>
    <i>Python package for fast multivariate polynomial interpolation in lp-type polynomial spaces.</i>
</p>

<p align="center">
    <a href="https://github.com/phil-hofmann/lpfun">
        <img src="https://img.shields.io/badge/GitHub-lpfun-blue?logo=github" alt="GitHub Repository">
    </a>
    <a href="./CITATION.cff">
        <img src="https://img.shields.io/badge/Cite-this%20repository-blue" alt="Cite this repo">
    </a>
    <a href="https://doi.org/10.48550/arXiv.2505.14909">
        <img src="https://img.shields.io/badge/arXiv-2505.14909-green.svg" alt="arXiv">
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
    <a href="https://choosealicense.com/licenses/mit/">
        <img src="https://img.shields.io/github/license/phil-hofmann/lpfun" alt="License">
    </a>
</p>

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Detailed example](#detailed-example)
- [API overview](#api-overview)
- [Citation](#citation)
- [Troubleshooting](#troubleshooting)
- [Team and Support](#team-and-support)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/phil-hofmann/lpfun.git
```

If you do not have `pip` installed, follow the official installation instructions:
https://pip.pypa.io/en/stable/installation/.

Optional environment setup references:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Poetry](https://python-poetry.org/docs/#installation)

## Quick start

```python
import numpy as np
from lpfun import Function

def f(x, y):
    return np.sin(x) * np.cos(y)

# Initialize the function space
fun = Function(
    spatial_dimension=2,
    polynomial_degree=10,
)

# Sample the function on the interpolation grid
values = f(fun.grid[:, 0], fun.grid[:, 1])

# Compute polynomial coefficients
coeffs = fun.interp(values)

# Reconstruct values on the interpolation grid
values_rec = fun.eval(coeffs)

# Differentiate with respect to the first coordinate
coeffs_dx = fun.diff(coeffs, dim=0, order=1)

# Evaluate the derivative on the interpolation grid
values_dx = fun.eval(coeffs_dx)
```

## Detailed example

The `Function` class provides fast interpolation, evaluation, differentiation,
and embedding between polynomial spaces.

```python
import time
import numpy as np

from lpfun import Function
from lpfun.basis.nodes import leja_nodes

# Initialize function space
fun = Function(
    spatial_dimension=3,
    polynomial_degree=20,
    nodes=leja_nodes,  # default nodes are cheb2nd_nodes
)

print(f"Dimension of the polynomial space = {len(fun)}")

def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)

# Sample function on the interpolation grid
values = f(fun.grid[:, 0], fun.grid[:, 1], fun.grid[:, 2])

# Interpolate
start = time.time()
coeffs = fun.interp(values)
print("fun.interp:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Reconstruct values on the interpolation grid
start = time.time()
values_rec = fun.eval(coeffs)
print("fun.eval:", "{:.2f}".format((time.time() - start) * 1000), "ms")

print(
    "max |values_rec - values| =",
    "{:.2e}".format(np.max(np.abs(values_rec - values))),
)

# Compute exact derivative with respect to z
def df_dz(x, y, z):
    return np.exp(z)

values_dz = df_dz(fun.grid[:, 0], fun.grid[:, 1], fun.grid[:, 2])

# Differentiate polynomial coefficients
start = time.time()
coeffs_dz = fun.diff(coeffs, dim=2, order=1)
print("fun.diff:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Reconstruct derivative values on the interpolation grid
values_dz_rec = fun.eval(coeffs_dz)

print(
    "max |values_dz_rec - values_dz| =",
    "{:.2e}".format(np.max(np.abs(values_dz_rec - values_dz))),
)

# Embed coefficients into a larger polynomial space
fun_larger = Function(
    spatial_dimension=3,
    polynomial_degree=30,
    nodes=leja_nodes,
    report=False,
)

embed_idx = fun.embed(fun_larger)

coeffs_larger = np.zeros(len(fun_larger))
coeffs_larger[embed_idx] = coeffs
```

When you run this code, you should see output similar to:

```text
---------------------+---------------------
                   Report
---------------------+---------------------
Spatial Dimension    | 3
Polynomial Degree    | 20
lp Degree            | 2.0
Condition V          | 1.44e+06
Amount of Coeffs     | 4_662
Construction         | 300.25 ms
Precompilation       | 19.80 ms
---------------------+---------------------

Dimension of the polynomial space = 4662
fun.interp: 0.55 ms
fun.eval: 0.53 ms
max |values_rec - values| = 8.88e-15
fun.diff: 0.19 ms
max |values_dz_rec - values_dz| = 5.80e-13
```

## API overview

The central object in `lpFun` is the `Function` class. It represents a
multivariate polynomial function space on a quasi-tensorial interpolation grid
and provides methods for interpolation, evaluation, differentiation, and
embedding.

### Construction

```python
from lpfun import Function

fun = Function(
    spatial_dimension=2,
    polynomial_degree=10,
    lp_degree=2.0,
)
```

### Main methods

| Task                               | Method                                | Description                                                                         |
| ---------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------- |
| Interpolate grid values            | `fun.interp(function_values)`         | Computes polynomial coefficients from values sampled on `fun.grid`.                 |
| Evaluate on the interpolation grid | `fun.eval(coefficients)`              | Reconstructs function values on the interpolation grid from coefficients.           |
| Evaluate at arbitrary points       | `fun(coefficients, points)`           | Evaluates the polynomial interpolant at user-specified points.                      |
| Differentiate                      | `fun.diff(coefficients, dim, order)`  | Computes coefficients of a partial derivative.                                      |
| Apply transposed derivative        | `fun.diffT(coefficients, dim, order)` | Applies the transpose of a partial differentiation operator.                        |
| Embed into a larger space          | `fun.embed(larger_fun)`               | Returns indices for embedding coefficients into a larger compatible function space. |

### Attributes

| Attribute               | Type            | Description                                                        |
| ----------------------- | --------------- | ------------------------------------------------------------------ |
| `fun.spatial_dimension` | `int`           | Spatial dimension `m`, i.e. the number of input variables.         |
| `fun.polynomial_degree` | `int`           | Maximum polynomial degree `n`.                                     |
| `fun.lp_degree`         | `float`         | Degree `p` of the l^p polynomial index set.                        |
| `fun.tube`              | `numpy.ndarray` | Directional polynomial degree constraints.                         |
| `fun.index_set`         | `numpy.ndarray` | Multi-index set defining the polynomial exponents.                 |
| `fun.nodes`             | `numpy.ndarray` | One-dimensional interpolation nodes.                               |
| `fun.grid`              | `numpy.ndarray` | Interpolation grid with shape `(fun.size, fun.spatial_dimension)`. |
| `fun.leja_order`        | `numpy.ndarray` | Leja ordering used to order the interpolation nodes.               |

### Special methods

| Expression     | Description                                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `len(fun)`     | Returns the dimension of the polynomial space, i.e. the number of coefficients.                                                |
| `fun == other` | Checks whether two `Function` objects describe the same polynomial space.                                                      |
| `repr(fun)`    | Returns a formatted setup report with dimension, polynomial degree, condition number, number of coefficients, and setup times. |

## Citation

[Download BibTeX](./lpFun.bib)

If you use lpFun in any public context, including publications, presentations,
or derivative software, please cite both the accompanying paper and the
software.

### Accompanying paper

Phil-Alexander Hofmann, Michael Hecht,
_Accelerating Multivariate Newton Interpolation in Downward Closed Polynomial Spaces_,
arXiv:2505.14909 [math.NA], 2026.
https://doi.org/10.48550/arXiv.2505.14909

### Software

Phil Hofmann. (2026). _lpFun_. GitHub.
https://github.com/phil-hofmann/lpFun

### Related references

- Michael Hecht, Phil-Alexander Hofmann, Damar Wicaksono, Uwe Hernandez Acosta, Krzysztof Gonciarz, Jannik Kissinger, Vladimir Sivkin, Ivo F. Sbalzarini, _Multivariate Newton interpolation in downward closed spaces reaches the optimal Bernstein–Walsh approximation rate_, IMA Journal of Numerical Analysis, draf137, 2026. https://doi.org/10.1093/imanum/draf137

- Damar Wicaksono, Uwe Hernandez Acosta, Sachin Krishnan Thekke Veettil, Jannik Kissinger, Michael Hecht, _Minterpy: multivariate polynomial interpolation in Python_, Journal of Open Source Software, 2025, 10(109):7702. https://doi.org/10.21105/joss.07702

- Damar Wicaksono et al. (2025). _Minterpy_. GitHub. https://github.com/minterpy-project/minterpy

## Troubleshooting

If you encounter segmentation faults, they are most likely caused by stale
[numba](https://numba.pydata.org/) cache files. This can happen when source
code changes invalidate previously compiled cache files.

### Option 1: Clear numba cache files

```bash
find src -name "*.nbi" -o -name "*.nbc" | xargs rm -f
```

### Option 2: Disable caching during development

```python
import lpfun

lpfun.CACHE = False
```

Set this _before_ importing any other `lpfun` modules.

## Team and Support

- [Phil-Alexander Hofmann](https://github.com/philippocalippo)
- [Piotr Held](https://github.com/qbrak)
- [Damar Wicaksono](https://github.com/damar-wicaksono)
- [Michael Hecht](https://github.com/mikeypice)

## License

The project is licensed under the [MIT License](LICENSE.txt).

## Acknowledgments

We deeply acknowledge:

- [Albert Cohen](https://scholar.google.com/citations?user=MkKZKAMAAAAJ)
- [Leslie Greengard](https://scholar.google.com/citations?user=lW-MnjMAAAAJ)
- [Oliver Sander](https://scholar.google.com/citations?user=UNhUp4kAAAAJ)
- [Peter Stadler](https://scholar.google.com/citations?user=pVnGRlkAAAAJ)
- [Shidong Jiang](https://scholar.google.com/citations?user=WFM32XwAAAAJ)
- [Uwe Hernandez Acosta](https://scholar.google.com/citations?user=8TUZP10AAAAJ)

and the support and resources provided by the
[Center for Advanced Systems Understanding](https://www.casus.science/)
([Helmholtz-Zentrum Dresden-Rossendorf](https://www.hzdr.de/)), where the
development of this project took place.
