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

## 📖 Citations

- CASUS. [Minterpy](https://github.com/casus/minterpy). 2024. Licensed under the [MIT](https://github.com/casus/minterpy/blob/main/LICENSE) License.
- philhofmann. [Bachelor thesis](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree). 2024. Licensed under the [MIT](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree/-/blob/main/LICENSE.txt?ref_type=heads) License.
- <a target="_blank" href="https://icons8.com/icon/WR2dlOv7LqTV/apple">Apple</a> Icon von <a target="_blank" href="https://icons8.com">Icons8</a>

## 💻 Installation

Please follow the steps below

1. Clone the project:

```bash
git clone https://github.com/philippocalippo/lpFun.git
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

## Usage

The `Transform` class can be used to perform forward and backward Newton transformations. Here is a basic example

```python
import time
import numpy as np
from lpfun import Transform

# Create a Transform object with dimension=3, degree=4
t = Transform(3, 20, 2)

# Warmup the JIT compiler
t.warmup()

# Print the dim of the polynomial space
print(f"N = {len(t)}")

# Define a function
def f(x, y, z):
    return np.sin(x) + np.cos(y) + np.exp(z)

# Calculate the exact function values
function_values = np.array([f(*x) for x in t.unisolvent_nodes])

# Perform the fast Newton transformation
start_time = time.time()
coeffs = t.push(function_values)
print(f"t.push: {int((time.time()-start_time)*1000)} ms")

# Perform the inverse fast Newton transformation
start_time = time.time()
reconstruction = t.pull(coeffs)
print(f"t.pull: {int((time.time()-start_time)*1000)} ms")

# Print the L1 norm of the difference between the reconstruction and the original function values
print(
    "max |reconstruction-function_values| = ",
    "{:.2e}".format(np.max(reconstruction - function_values)),
)

# Evaluate at single point
x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
fx = f(*x)
start_time = time.time()
reconstruction_fx = t.eval(coeffs, x)
print(f"t.eval: {int((time.time()-start_time)*1000)} ms")

# Print the absolute error
print("|reconstruction_fx-fx| = ", "{:.2e}".format(np.abs(fx - reconstruction_fx)))

# Define the derivative
def dx_f(x, y, z):
    return np.zeros_like(x) + np.zeros_like(y) + np.exp(z)

# Calculate the exact derivative
dx_function_values = np.array([dx_f(*x) for x in t.unisolvent_nodes])

# Perform the derivative fast Newton transformation
start_time = time.time()
dx_coeffs = t.dx(coeffs, 2)
print(f"t.dx: {int((time.time()-start_time)*1000)} ms")

# Perform the inverse fast Newton transformation
dx_reconstruction = t.pull(dx_coeffs)

# Print the L1 norm of the difference between the reconstruction of the derivative and the original derivative function values
print(
    "max |dx_reconstruction-dx_function_values| = ",
    "{:.2e}".format(np.max(np.abs(dx_reconstruction - dx_function_values))),
)
```

When you run this code, you should see an output similar to

```
N = 4662
t.push: 11 ms
t.pull: 10 ms
max |reconstruction-function_values| =  2.53e-14
t.eval: 0 ms
|reconstruction_fx-fx| =  8.88e-15
t.dx: 1 ms
max |dx_reconstruction-dx_function_values| =  6.03e-12
```

## Acknowledgments

I would like to acknowledge:

- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Dr. Damar Wicakson](https://www.casus.science/de-de/team-members/dr-damar-wicaksono/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Prof. Dr. Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
