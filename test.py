import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

from util import norm
from consts import MDSISDLineParam, TDSISDFixedPhiParam

# G = MDSISDLineParam(1, 1.2, 4)
g = TDSISDFixedPhiParam(1, 1.2)
# t, _, x2, xi = g.traj_r(4, .8)
# x1, x2, xi, xc = g.point_on_barrier(7, 4, .8)
x1, x2, xi, xc = g.barrier_t(9, 3.6)
xn = g.barrier_n(9, 3.6)

# print(x1)
plt.plot(xi[:,0], xi[:,1])
plt.plot(-xi[:,0], xi[:,1])
plt.plot(xn[:,0], xn[:,1])
plt.plot(-xn[:,0], xn[:,1])
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





# g = MDSISDLineParam(1, 1.5, 5)

# b = np.linspace(pi/2, g.beta, 20)
# t = [g.t1(bb) for bb in b]
# plt.plot(t, b)
# plt.grid()
# plt.show()

# ys = np.linspace(g.r, g.ymax(), 10)

# t = g.tmax()

# xds_l, xds_r = [], []
# for y in ys:
# 	reg = g.partition(y, t)
# 	if reg == 'natural':
# 		xd_l, xd_r = g.xd_natural(y, t)
# 	if reg == 'envelope':
# 		xd_l, xd_r = g.xd_envelope(y, t)
# 	xds_l.append(xd_l)
# 	xds_r.append(xd_r)

# plt.plot(xds_l, ys)
# plt.plot(xds_r, ys)
# plt.gca().axis('equal')

# t = 0.9*t
# xds_l, xds_r = [], []
# for y in ys:
# 	reg = g.partition(y, t)
# 	if reg == 'natural':
# 		xd_l, xd_r = g.xd_natural(y, t)
# 	if reg == 'envelope':
# 		xd_l, xd_r = g.xd_envelope(y, t)
# 	xds_l.append(xd_l)
# 	xds_r.append(xd_r)

# plt.plot(xds_l, ys)
# plt.plot(xds_r, ys)
# plt.gca().axis('equal')

# plt.show()


# y = 1.3
# tmin = g.tmin(y)
# tmax = g.tmax()
# xds_l, xds_r = [], []
# ts = np.linspace(tmin, tmax, 10)
# for t in ts:
# 	reg = g.partition(y, t)
# 	if reg == 'natural':
# 		xd_l, xd_r = g.xd_natural(y, t)
# 	if reg == 'envelope':
# 		xd_l, xd_r = g.xd_envelope(y, t)
# 	xds_l.append(xd_l)
# 	xds_r.append(xd_r)

# for xd_l, xd_r, t in zip(xds_l, xds_r, ts):
# 	plt.plot([xd_l, xd_r], [t, t])

# plt.show()


# p = []
# for b in np.linspace(pi/2, 0, 25):
# 	p.append(g.envelope(b))
# p = np.asarray(p)
# plt.plot(p[:,0], p[:,1])

# pp = []
# for b in np.linspace(pi/2, g.beta, 20):
# 	pp.append(g.involute(pi/2, b))
# for gm in np.linspace(g.gamma+.1, pi/2, 20):
# 	pp.append(g.involute(pi/2, g.beta-.1, gm))
# pp = np.asarray(pp)
# plt.plot(pp[:,0], pp[:,1])

# pp = []
# for b in np.linspace(pi/3, g.beta, 20):
# 	pp.append(g.involute(pi/3, b))
# for gm in np.linspace(g.gamma+.1, pi/2, 20):
# 	pp.append(g.involute(pi/3, g.beta-.1, gm))		
# pp = np.asarray(pp)
# plt.plot(pp[:,0], pp[:,1])

# plt.plot(g.xi_b, g.yi_b, 'o')
# plt.plot(g.xd_b, 0, 'x')

# plt.grid()
# plt.axis('equal')

# plt.show()

