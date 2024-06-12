# NewFun

A package which uses Newton polynomials for function approximation.

## Authors

- [Phil-Alexander Hofmann](https://gitlab.com/philippo_calippo) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/))

## 📜 License

The project is licensed under the [MIT License](LICENSE.txt).

## 📖 Citations

- CASUS. [Minterpy](https://github.com/casus/minterpy). 2024. Licensed under the [MIT](https://github.com/casus/minterpy/blob/main/LICENSE) License.
- philhofmann. [Bachelor thesis](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree). 2024. Licensed under the [MIT](https://gitlab.com/philhofmann/implementation-and-complexity-analysis-of-algorithms-for-multivariate-newton-polynomials-of-p-degree/-/blob/main/LICENSE.txt?ref_type=heads) License.
- <a target="_blank" href="https://icons8.com/icon/WR2dlOv7LqTV/apple">Apple</a> Icon von <a target="_blank" href="https://icons8.com">Icons8</a>

## 💻 Installation

Please follow the steps below:

1. Clone the project:

```bash
git clone https://gitlab.com/philhofmann/newfun.git
```

2. Create a virtual environment:

```bash
conda env create -f environment.yml
```

3. Activate environment:

```bash
conda activate newfun
```

4. Install using pip:

```bash
pip install -e .
```

5. Run the tests to verify the installation:

```bash
pytest
```

6. If you want to deactivate the environment:

```bash
conda deactivate
```

## Usage

The `Transform` class can be used to perform forward and backward Newton transformations. Here is a basic example:

```python
from newfun import Transform
import numpy as np

# Create a Transform object with dimension=3, degree=4
t = Transform(3, 4)

# Print the dim of the polynomial space
print(f"N = {len(t)}")

# Define a function
def f(x, y, z):
    return x**2 + 2*x**3 + y + y**2 + z + z**2 + 3*z**4

# Generate function values from the unisolvent nodes
function_values = [f(*x) for x in t.unisolvent_nodes]

# Perform the fast Newton transformation
coeffs = t.fnt(function_values)

# Perform the inverse fast Newton transformation
reconstruction = t.ifnt(coeffs)

# Print the L1 norm of the difference between the reconstruction and the original function values
print(f"||reconstruction-function_values||_1: {np.linalg.norm(reconstruction-function_values)}")
```

When you run this code, you should see the following output:

```
N = 54
||reconstruction-function_values||_1: 7.685725819700564e-15
```

## Acknowledgments

I would like to acknowledge:

- [Prof. Dr. Peter Stadler](https://www.uni-leipzig.de/personenprofil/mitarbeiter/prof-dr-peter-florian-stadler) - [University of Leipzig](https://www.uni-leipzig.de/),
- [Prof. Dr. Michael Hecht](https://www.casus.science/de-de/team-members/michael-hecht/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),
- [Dr. Damar Wicakson](https://www.casus.science/de-de/team-members/dr-damar-wicaksono/) - [CASUS](https://www.casus.science/) ([HZDR](https://www.hzdr.de/)),

and the support and resources provided by the [Center for Advanced Systems Understanding](https://www.casus.science/) ([Helmholtz-Zentrum Dreden-Rossendorf](https://www.hzdr.de/)) where the development of this project took place.
