__version__ = "0.0.1"
import numpy as np
import numba as nb

### CONFIGURATION OPTIONS

NP_INT = np.int64
NB_INT = nb.int64
NP_FLOAT = np.float64
NB_FLOAT = nb.float64

PARALLEL = True
WARMUP = True
MESSAGES = True

THRESHOLD = 20_000_000

###

from .transform import Transform
