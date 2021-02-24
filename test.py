import numpy as np
from math import pi, cos, sin

import matplotlib.pyplot as plt

# from base_2dsir0 import TDSISDPointCap
import cvxpy as cp

P1 = np.array([[1, 0, 0, 0],
			   [0, 1, 0, 0],
			   [0, 0, 0, 0],
			   [0, 0, 0, 0]])		# maximize v1y + v2y

v = np.array([1, 2, 3, 4])

print(cp.quad_form(v, P1).value)