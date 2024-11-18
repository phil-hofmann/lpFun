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
