import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

from util import norm
from consts import MDSISDLineParam, TDSISDFixedPhiParam

c = ['k', 'b', 'r', 'm', 'y', 'g', 'c']

g = TDSISDFixedPhiParam(1, 1.4)

L = 10
# tmin, tmax = g.get_trange_fromL(L)
# tstar = g.barrier_e_tmin(L)
# print('trange: ', tmin, tmax, tstar)
# # dt = tmax - tmin
# # tstar = g.barrier_e_tmin(L)

# for i, t in enumerate(np.linspace(tstar, tmax, 20)):

# 	xi, xc = g.barrier(L, t)

# 	plt.plot(xi[:,0], xi[:,1], color=c[i%7])
# 	plt.plot(xc[:,0], xc[:,1], color=c[i%7])

# plt.grid()
# plt.axis('equal')
# plt.show()

tmin, tmax = g.get_trange_fromL(L)
t = 0.5*(tmin + tmax)
dt = 0.2

xi, xc = g.barrier(L, t)
plt.plot(xi[:,0], xi[:,1], color=c[0])
plt.plot(xc[:,0], xc[:,1], color=c[0])

for i in range(12):
	L = L - 2*dt
	t = t - dt
	xi, xc = g.barrier(L, t)
	plt.plot(xi[:,0], xi[:,1], color=c[(i+1)%7])
	plt.plot(xc[:,0], xc[:,1], color=c[(i+1)%7])

plt.grid()
plt.axis('equal')
plt.show()