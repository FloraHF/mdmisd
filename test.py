import numpy as np
from math import pi, cos, sin
from sklearn import linear_model

import matplotlib.pyplot as plt

from bases.base_2dsihg import TDSISDHagdorn

a = np.linspace(1, 10, 4)
aa = np.vstack([a] + [np.zeros(4), np.ones(4)])
print(aa)