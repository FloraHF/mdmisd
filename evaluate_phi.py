import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

from util import norm
from consts import MDSISDLineParam, TDSISDFixedPhiParam

# G = MDSISDLineParam(1, 1.2, 4)
g = TDSISDFixedPhiParam(1, 1.2)
# t, _, x2, xi = g.traj_r(4, .8)
# x1, x2, xi, xc = g.point_on_barrier(7, 4, .8)
x1, x2, xi, xc = g.barrier_t(9, 4)

# print(x1)
plt.plot(xi[:,0], xi[:,1])
plt.plot(-xi[:,0], xi[:,1])
plt.plot(xc[:,0], xc[:,1])
plt.plot(-xc[:,0], xc[:,1])
plt.plot(x1[0,0], x1[0,1], 'o')
plt.plot(x2[0,0], x2[0,1], 'o')

x = [x1[0,0]+cos(t) for t in np.linspace(0, 2*pi, 50)]
y = [x1[0,1]+sin(t) for t in np.linspace(0, 2*pi, 50)]
plt.plot(x, y)
plt.grid()


# plt.plot([x1[0], x2[0]], [x1[1], x2[1]])
# plt.plot(xi[0], xi[1], 'o')
# plt.plot(xc[0], xc[1], 'x')
plt.axis('equal')
plt.show()