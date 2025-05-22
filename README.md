<p align="center">
  <img src="docs/social-banner-bg-rounded.png" height="128" width="384"/>
</p>
<p align="center">
    A package for fast multivariate interpolation and differentiation in downward closed polynomial spaces.
</p>

[![Cite this repo](https://img.shields.io/badge/Cite-this%20repository-blue)](https://github.com/phil-hofmann/lpFun/blob/main/CITATION.cff)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14909-green.svg)](https://arxiv.org/abs/2505.14909)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/minterpy-project/minterpy)](https://choosealicense.com/licenses/mit/)

## 👥 Team

- [Phil-Alexander Hofmann](https://gitlab.com/philippo_calippo) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Damar Wicaksono](https://gitlab.com/damar-wicaksono) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Michael Hecht](https://sites.google.com/view/prof-dr-michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)).

## 💬 Citation

> 📚 [Download BibTeX](./lpFun.bib) (The project is licensed under the [MIT License](LICENSE.txt).)

If you use lpFun in any public context (publications, presentations, or derivative software), **please cite both**:

- **Accompanying paper**:

  > Phil-Alexander Hofmann, Damar Wicaksono, Michael Hecht.
  > _Fast Newton Transform: Interpolation in Downward Closed Polynomial Spaces_.  
  > arXiv, 2025. https://arxiv.org/abs/2505.14909

- **Software**:
  > Phil-Alexander Hofmann. _lpFun_. 2025.  
  > https://github.com/phil-hofmann/lpFun

<br/>

**Related references**:

- _Hecht et al._ Multivariate Newton Interpolation in Downward Closed Spaces Reaches the Optimal Geometric Approximation Rates for Bos–Levenberg–Trefethen Functions. arXiv, 2025. https://arxiv.org/pdf/2504.17899
- _Damar Wicaksono et al._ Minterpy: multivariate polynomial interpolation in
  Python. The Journal of Open Source Software, 2025. https://joss.theoj.org/papers/10.21105/joss.07702
- _Phil-Alexander Hofmann._ Implementation and Complexity Analysis of Algorithms for Multivariate Newton Polynomials of p Degree. 2024.
- Damar Wicaksono et al. _Minterpy_. 2025. https://github.com/minterpy-project/minterpy
- Phil-Alexander Hofmann. _Prototype of lpFun_. 2024. https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree

## 💻 Installation

### 1. Including in Your Project

If you want to include this package in your project, you can install it directly from the GitHub repository:

1. Create a conda environment with Python 3.9

```bash
conda create --name myenv python=3.9.20
```

2. Activate environment

```bash
conda activate myenv
```

3. Install the package from the GitHub repository

```bash
pip install git+https://github.com/phil-hofmann/lpfun.git
```

4. If you want to deactivate the environment

```bash
conda deactivate
```

### 2. Setting Up the Repository on Your Local Machine

Please follow the steps below

1. Clone the project

```bash
git clone https://github.com/phil-hofmann/lpfun.git
```

2. Create a conda environment

```bash
conda env create -f environment.yml
```

3. Activate environment

```bash
conda activate lpfun
```

4. Install lpfun package using pip

```bash
pip install -e .
```

5. Run the tests to verify the installation

```bash
pytest -v
```

6. If you want to deactivate the environment

```bash
conda deactivate
```

## 📖 Tutorial

[tutorial.py](docs/tutorial.py)

### 1. Short Version

```python
import numpy as np
from lpfun import Transform

# from lpfun.utils import leja_nodes # NOTE optional


# Function f to approximate
def f(x, y):
    return np.sin(x) * np.cos(y)


# Initialise Transform object
t = Transform(
    spatial_dimension=2,
    polynomial_degree=10,
    # nodes=leja_nodes # NOTE optional
)

# Compute function values on the grid
values_f = f(t.grid[:, 0], t.grid[:, 1])

# Perform the fast Newton transform (FNT) to compute the coefficients
coeffs_f = t.fnt(values_f)

# Compute the approximate first derivative in the first coordinate direction
coeffs_df = t.dx(coeffs_f, i=0, k=1)

# Evaluate the approximate derivative at random points
random_points = np.random.rand(10, 2)
random_df = t.eval(coeffs_df, random_points)

# Perform the inverse Newton transform (IFNT) to evaluate the approximate derivative on the grid
rec_df = t.ifnt(coeffs_df)
```

### 2. Long Version

The `Transform` class enables forward and backward transform, as well as the computation of derivatives and their adjoints.

```python
import time
import numpy as np
from lpfun import Transform
from lpfun.utils import leja_nodes

# Initialise Transform object
t = Transform(
    spatial_dimension=3,
    polynomial_degree=20,
    nodes=leja_nodes,  # NOTE default nodes are cheb2nd_nodes
)

# Dimension of the polynomial space
print(f"N = {len(t)}")


# Function f to approximate
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)


# Compute function values on the grid
values_f = f(t.grid[:, 0], t.grid[:, 1], t.grid[:, 2])

# Perform the fast Newton transform (FNT) to compute the coefficients
start = time.time()
coeffs_f = t.fnt(values_f)
print("t.fnt:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Perform the inverse fast Newton transform (IFNT) to reconstruct values on the grid
start = time.time()
rec_f = t.ifnt(coeffs_f)
print("t.ifnt:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Measure the maximum norm error for reconstruction
print(
    "max |rec_f - values_f| =",
    "{:.2e}".format(np.max(np.abs(rec_f - values_f))),
)

# Evaluate approximate f at random points
rand_points = np.random.rand(10, 3)
f_rand = f(rand_points[:, 0], rand_points[:, 1], rand_points[:, 2])
start = time.time()
rec_f_rand = t.eval(coeffs_f, rand_points)
print("t.eval (random points):", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Print the maximum norm error
print(
    "|f_random - rec_f_random| =", "{:.2e}".format(np.max(np.abs(f_rand - rec_f_rand)))
)


# Compute the exact derivative
def df_dz(x, y, z):
    return np.exp(z)


# Compute exact derivative values on the grid
values_df = df_dz(t.grid[:, 0], t.grid[:, 1], t.grid[:, 2])

# Compute approximate derivative coefficients
start = time.time()
coeffs_df = t.dx(coeffs_f, i=2, k=1)
print("t.dx:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Perform inverse transform on derivative coefficients to reconstruct values on grid
rec_df = t.ifnt(coeffs_df)

# Print maximum norm error
print(
    "max |rec_df - values_df| =",
    "{:.2e}".format(np.max(np.abs(rec_df - values_df))),
)

# Evaluate approximate df at random points
df_rand = df_dz(rand_points[:, 0], rand_points[:, 1], rand_points[:, 2])
start = time.time()
rec_df_rand = t.eval(coeffs_df, rand_points)
print("t.eval (random points):", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Print the maximum norm error
print(
    "max |df_rand-rec_df_rand| =",
    "{:.2e}".format(np.max(np.abs(df_rand - rec_df_rand))),
)

# Embed the approximate derivative into a bigger polynomial space space
t_prime = Transform(
    spatial_dimension=3,
    polynomial_degree=30,
    nodes=leja_nodes, # NOTE default nodes are cheb2nd_nodes
    report=False,
)
phi = t.embed(t_prime)
coeffs_df_prime = np.zeros(len(t_prime))
coeffs_df_prime[phi] = coeffs_df.copy()
```

When you run this code, you should see outputs similar to:

```
---------------------+---------------------
                   Report
---------------------+---------------------
Spatial Dimension    | 3
Polynomial Degree    | 20
lp Degree            | 2.0
Condition V          | 1.44e+06
Amount of Coeffs     | 4_662
Construction         | 2_303.00 ms
Precompilation       | 16_603.96 ms
---------------------+---------------------

N = 4662
t.fnt: 0.59 ms
t.ifnt: 0.52 ms
max |rec_f - values_f| = 7.11e-15
t.eval (random points): 0.19 ms
|f_random - rec_f_random| = 8.44e-15
t.dx: 0.50 ms
max |rec_df - values_df| = 1.03e-12
t.eval (random points): 0.19 ms
max |df_rand-rec_df_rand| = 5.15e-14
```

## Acknowledgments

We deeply acknowledge:

- [Albert Cohen](https://www.ljll.fr/en/membre/cohen-albert/) - [Sorbonne Universit&eacute;](https://www.sorbonne-universite.fr/),
- [Leslie Greengard](https://www.simonsfoundation.org/people/leslie-greengard/) - [Flatiron Institute](https://www.simonsfoundation.org/flatiron/),
- [Oliver Sander](https://tu-dresden.de/mn/math/numerik/sander) - [Technical University of Dresden](https://tu-dresden.de/),
- [Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),
- [Shidong Jiang](https://www.simonsfoundation.org/people/shidong-jiang/) - [Flatiron Institute](https://www.simonsfoundation.org/flatiron/),
- [Uwe Hernandez Acosta](https://github.com/szabo137) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
