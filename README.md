# lpFun

<p align="center">
  <img src="social-banner-bg-rounded.png" height="128" width="384"/>
</p>
<p align="center">
    A package for fast multivariate interpolation and spectral methods in lower spaces.
</p>

## Authors

- [Phil-Alexander Hofmann](https://gitlab.com/philippo_calippo) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/))

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 💬 Citations

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
pip install git+https://github.com/philippocalippo/lpfun.git
```

4. If you want to deactivate the environment

```bash
conda deactivate
```

### Setting Up the Repository on Your Local Machine

Please follow the steps below

1. Clone the project

```bash
git clone https://github.com/philippocalippo/lpfun.git
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
pytest
```

6. If you want to deactivate the environment

```bash
conda deactivate
```

## 📖 Tutorial :: Short Version

```python
import numpy as np
from lpfun import Transform

# Initialise Transform object
t = Transform(spatial_dimension=3, polynomial_degree=20)


# Function values
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)


fx = np.array([f(*x) for x in t.grid])

# Perform the fast Newton transform (FNT)
coeffs = t.fnt(fx)

# Compute the approximate derivative
coeffs = t.dx(coeffs, 2)

# Perform the inverse Newton transform (IFNT)
rec = t.ifnt(coeffs)

# Evaluate at a single point
x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
rec = t.eval(coeffs, x)
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


# Function values
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)


fx = np.array([f(*x) for x in t.grid])

# Perform the fast Newton transform (FNT)
start = time.time()
coeffs = t.fnt(fx)
print("t.fnt:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Perform the inverse fast Newton transform (IFNT)
start = time.time()
rec = t.ifnt(coeffs)
print("t.ifnt:", "{:.2f}".format((time.time() - start) * 1000), "ms")

# Measure the maximum norm error
print(
    "max |rec-fx| =",
    "{:.2e}".format(np.max(rec - fx)),
)

# Evaluate at a single point
x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
fx = f(*x)
start = time.time()
rec = t.eval(coeffs, x)
print("t.eval:", "{:.2f}".format((time.time() - start) * 1000), "ms")
print("|rec-fx| =", "{:.2e}".format(np.abs(fx - rec)))


# Compute the exact derivative
def df_dz(x, y, z):
    return np.zeros_like(x) + np.zeros_like(y) + np.exp(z)


dfx = np.array([df_dz(*x) for x in t.grid])

# Compute the approximate derivative
start_time = time.time()
coeffs = t.dx(coeffs, 2)
print(f"t.dx:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Perform the inverse Newton transform (IFNT)
rec = t.ifnt(coeffs)

# Print the maximum norm error
print(
    "max |dx_reconstruction-dx_function_values| =",
    "{:.2e}".format(np.max(np.abs(rec - dfx))),
)
```

When you run this code, you should see an output similar to

```
N = 4662
t.fnt: 12.38 ms
t.ifnt: 13.05 ms
max |rec-fx| = 1.73e-14
t.eval: 1.17 ms
|rec-fx| = 1.51e-14
t.dx: 0.48 ms
max |dx_reconstruction-dx_function_values| = 2.20e-11
```

## Acknowledgments

I would like to acknowledge:

- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Dr. Damar Wicaksono](https://www.casus.science/de-de/team-members/dr-damar-wicaksono/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
