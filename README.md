# Revolutionizing Function Approximation and Spectral Methods in Arbitrary Dimensions!
<p align="center">
  <img src="social-banner-bg-rounded.png" height="128" width="384"/>
</p>
<p align="center">
    A package which uses l^p degree polynomials for function approximation and differentiation.
</p>

## Authors

- [Phil-Alexander Hofmann](https://gitlab.com/philippo_calippo) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/))

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 💬 Citations

- CASUS. [Minterpy](https://github.com/casus/minterpy). 2024. Licensed under the [MIT](https://github.com/casus/minterpy/blob/main/LICENSE) License.
- philhofmann. [Bachelor thesis](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree). 2024. Licensed under the [MIT](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree/-/blob/main/LICENSE.txt?ref_type=heads) License.

## 💻 Installation

Please follow the steps below

1. Clone the project:

```bash
git clone https://github.com/philippocalippo/lpfun.git
```

2. Create a virtual environment

```bash
conda env create -f environment.yml
```
 
3. Activate environment:

```bash
conda activate lpfun
```

4. Install using pip

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

## 📖 Usage

The `Transform` class can be used to perform forward and backward l^p transformations and derivatives.

```python
import time
import numpy as np
from lpfun import Transform

# Create a Transform object with spatial_dimension=3, polynomial_degree=4, p=2 (default value), mode="newton" (default value)
t = Transform(spatial_dimension=3, polynomial_degree=20)

# Warmup the JIT compiler
t.warmup()

# Print the dimension of the polynomial space
print(f"N = {len(t)}")

# Define a function
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)

# Calculate the exact function values on the unisolvent nodes
function_values = np.array([f(*x) for x in t.unisolvent_nodes])

# Perform the forward transformation
start_time = time.time()
coeffs = t.push(function_values)
print("t.push:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Perform the backward transformation
start_time = time.time()
reconstruction = t.pull(coeffs)
print("t.pull:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Print the maximum norm error
print(
    "max |reconstruction-function_values| =",
    "{:.2e}".format(np.max(reconstruction - function_values)),
)

# Evaluate at a single point
x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
fx = f(*x)
start_time = time.time()
reconstruction_fx = t.eval(coeffs, x)
print("t.eval:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Print the absolute error
print("|reconstruction_fx-fx| =", "{:.2e}".format(np.abs(fx - reconstruction_fx)))

# Define the derivative
def dx_f(x, y, z):
    return np.zeros_like(x) + np.zeros_like(y) + np.exp(z)

# Calculate the exact derivative dx_3 on the unisolvent nodes
dx_function_values = np.array([dx_f(*x) for x in t.unisolvent_nodes])

# Compute the derivative dx_3
start_time = time.time()
dx_coeffs = t.dx(coeffs, 2)
print(f"t.dx:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Perform the backward transformation
dx_reconstruction = t.pull(dx_coeffs)

# Print the maximum norm error
print(
    "max |dx_reconstruction-dx_function_values| =",
    "{:.2e}".format(np.max(np.abs(dx_reconstruction - dx_function_values))),
)
```

When you run this code, you should see an output similar to

```
N = 4662
t.push: 9.09 ms
t.pull: 8.67 ms
max |reconstruction-function_values| = 2.53e-14
t.eval: 0.62 ms
|reconstruction_fx-fx| = 9.33e-15
t.dx: 1.58 ms
max |dx_reconstruction-dx_function_values| = 5.54e-12
```

In Lagrangian basis it is no possible to perform such l^p differentiation directly for the non-tensorial case as demonstrated above. Nevertheless, this feature is implemented for the special case where p = infinity (or spatial_dimension = 1).

```python
import time
import numpy as np
from lpfun import Transform

# Create a Transform object with dimension=3, polynomial_degree=4, p=np.infty, mode="lagrange"
t = Transform(spatial_dimension=3, polynomial_degree=20, p=np.infty, mode="lagrange")

# Warmup the JIT compiler
t.warmup()

# Print the dimension of the polynomial space
print(f"N = {len(t)}")

# Define a function
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)

# Calculate the exact function values on the unisolvent nodes
function_values = np.array([f(*x) for x in t.unisolvent_nodes])

# Define the derivative
def dx_f(x, y, z):
    return np.zeros_like(x) + np.zeros_like(y) + np.exp(z)

# Calculate the exact derivative dx_3 on the unisolvent nodes
dx_function_values = np.array([dx_f(*x) for x in t.unisolvent_nodes])

# Compute the derivative dx_3
start_time = time.time()
dx_reconstruction = t.dx(function_values, 2)
print(f"t.dx:", "{:.2f}".format((time.time() - start_time) * 1000), "ms")

# Print the maximum norm error
print(
    "max |dx_reconstruction-dx_function_values| =",
    "{:.2e}".format(np.max(np.abs(dx_reconstruction - dx_function_values))),
)
```

When you run this code, you should see an output similar to

```
N = 9261
t.dx: 0.17 ms
max |dx_reconstruction-dx_function_values| = 3.65e-13
```

The little Lagrangian example from above is also showcasing that N is much bigger for p being infinity instead of two!
Obviously 0.17ms is still much less than the Newton mode offers. This is an artifact of NumPys BLAS Library which is highly optimized and wont appear for higher dimensional or polynomial degree applications.

## Acknowledgments

I would like to acknowledge:

- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Dr. Damar Wicaksono](https://www.casus.science/de-de/team-members/dr-damar-wicaksono/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
