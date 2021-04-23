import numpy as np
import matplotlib.pyplot as plt

from base_2dsihg import TDSISDHagdorn
from util import plot_cap_range, norm

g = TDSISDHagdorn(1, 1, 1.3)


L = 3
tmin = g.tmin(L)
tmax = g.tmax(L)
# t = tmin + k*(tmax - tmin)


for k in [.1, .5, .9]:
	t = tmin + k*(tmax - tmin)
	xi_e = g.isoc_e(L, t)
	xi_n = g.isoc_n(L, t)
	plt.plot(xi_e[:,0], xi_e[:,1])
	plt.plot(xi_n[:,0], xi_n[:,1])



plt.plot([-L/2], [0], 'bo')
plt.plot([ L/2], [0], 'bo')
plot_cap_range(plt.gca(), np.array([-L/2, 0]), 1)
plot_cap_range(plt.gca(), np.array([ L/2, 0]), 1)

plt.axis('equal')
plt.show()



# x1s, xis = [], []
# for dphi in np.linspace(0, 2, 20):
# 	x, t = g.sdsi_barrier_e(g.lb+dphi, 0)
# 	x1 = x[:2]
# 	xi = x[2:]
# 	x1s.append(x1)
# 	xis.append(xi)
# 	print(norm(x1 - xi))

# x1s = np.asarray(x1s)
# xis = np.asarray(xis)
# plt.plot(xis[:,0], xis[:,1])
# plt.plot(x1s[:,0], x1s[:,1])
# plt.axis('equal')
# plt.show()