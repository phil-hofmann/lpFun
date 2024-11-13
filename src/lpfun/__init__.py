__version__ = "0.0.1"
import numpy as np
import numba as nb

NP_INT = np.int64
NB_INT = nb.int64
NP_FLOAT = np.float64
NB_FLOAT = nb.float64
EXPENSIVE = 20_000_000
PARALLEL = False
from .transform import Transform
