# lpFun

<p align="center">
  <img src="social-banner-bg-rounded.png" height="128" width="384"/>
</p>
<p align="center">
    A package for fast multivariate interpolation and differentiation in downward closed polynomial spaces.
</p>

## Authors

- [Phil-Alexander Hofmann](https://gitlab.com/philippo_calippo) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/))

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 💬 Citations

**Please cite the following work when using this framework in any public context**:
_Fast Newton Transform: Interpolation in Downward Closed Polynomial Spaces_ [https://arxiv.org/]

**Related references**:

- Multivariate Newton Interpolation in Downward Closed Spaces Reaches the Optimal Geometric Approximation Rates for Bos–Levenberg–Trefethen Functions [https://arxiv.org/pdf/2504.17899]
- CASUS. [Minterpy](https://github.com/casus/minterpy). 2024. Licensed under the [MIT](https://github.com/casus/minterpy/blob/main/LICENSE) License.
- philhofmann. [Bachelor thesis](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree). 2024. Licensed under the [MIT](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree/-/blob/main/LICENSE.txt?ref_type=heads) License.

## 💻 Installation

### Including in Your Project

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

### Setting Up the Repository on Your Local Machine

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

## 📖 Tutorial : Short Version

```python
import numpy as np
from lpfun import Transform

# Function f to approximate
def f(x, y):
    return np.sin(x) * np.cos(y)

# Initialise Transform object
t = Transform(spatial_dimension=2, polynomial_degree=10)

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

## 📖 Tutorial

The `Transform` class enables forward and backward transform, as well as the computation of derivatives and their adjoints.

```python
import time
import numpy as np
from lpfun import Transform

# Initialise Transform object
t = Transform(spatial_dimension=3, polynomial_degree=20)

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
t_prime = Transform(spatial_dimension=3, polynomial_degree=30, report=False)
phi = t_prime.embed(t)
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
Condition V          | 1.33e+06
Amount of Coeffs     | 4_662
Construction         | 2_113.15 ms
Precompilation       | 17_745.04 ms
---------------------+---------------------

N = 4662
t.fnt: 0.60 ms
t.ifnt: 0.64 ms
max |rec_f - values_f| = 8.44e-15
t.eval (random points): 0.23 ms
|f_random - rec_f_random| = 9.77e-15
t.dx: 0.54 ms
max |rec_df - values_df| = 1.72e-12
t.eval (random points): 0.18 ms
max |df_rand-rec_df_rand| = 1.22e-13
```

## Acknowledgments

I would like to acknowledge:

- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Dr. Damar Wicaksono](https://www.casus.science/de-de/team-members/dr-damar-wicaksono/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),
- [Prof. Dr. Oliver Sander](https://tu-dresden.de/mn/math/numerik/sander) - [Techincal University of Dresden](https://tu-dresden.de/),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
