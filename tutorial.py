### Tutorial : Short Version

# import numpy as np
# from lpfun import Transform

# # Function f to approximate
# def f(x, y):
#     return np.sin(x) * np.cos(y)

# # Initialise Transform object
# t = Transform(spatial_dimension=2, polynomial_degree=10)

# # Compute function values on the grid
# values_f = f(t.grid[:, 0], t.grid[:, 1])

# # Perform the fast Newton transform (FNT) to compute the coefficients
# coeffs_f = t.fnt(values_f)

# # Compute the approximate first derivative in the first coordinate direction
# coeffs_df = t.dx(coeffs_f, i=0, k=1)

# # Evaluate the approximate derivative at random points
# random_points = np.random.rand(10, 2)
# random_df = t.eval(coeffs_df, random_points)

# # Perform the inverse Newton transform (IFNT) to evaluate the approximate derivative on the grid
# rec_df = t.ifnt(coeffs_df)


### Tutorial

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
