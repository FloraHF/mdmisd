import os

import numpy as np
from math import pi, cos, sin, log, sqrt
from scipy import interpolate
import pandas as pd

import matplotlib.pyplot as plt

# import cv2

from util import norm
from base_2dsi import TDSISDFixedPhiParam

c = ['k', 'b', 'r', 'm', 'y', 'g', 'c']

def get_barrier(L, ls='-'):
	g = TDSISDFixedPhiParam(1, 1.4)
	tmin, tmax = g.get_trange_fromL(L)
	tstar = g.barrier_e_tmin(L)
	print(tmin, tmax, tstar)

	plt.figure()

	tht = np.linspace(0, 2*pi, 50)
	xd = np.array([L/2 + g.r*cos(t) for t in tht])
	yd = np.array([	 	g.r*sin(t) for t in tht])
	plt.plot(xd, yd, color='k', ls=ls)

	fs, xm = [], []

	# ts = np.linspace(tstar, tmax, 10)
	ts = [tstar]
	dt = 0.05
	base = 1.1
	k = 0.
	while ts[-1] < tmax:
		ts.append(min(ts[-1] + base**k*dt, tmax))
		base *= 1.1
		k += 1.1
	ts = ts[:0] + [0.5*ts[0] + 0.5*ts[1]] + ts[1:]

	path = 'bdata/a_%.6f/r_%.6f/L_%.6f/'%(g.a, g.r, L)
	if not os.path.exists(path):
		os.makedirs(path)

	# compute 
	for i, t in enumerate(ts):
		print(t)
		xi, xc, xe, v1, v2, vi = g.barrier(L, t)

		if i == 0:
			fs.append(interpolate.interp1d(xi[:,0], xi[:,1],
											fill_value="extrapolate"))
			xm.append(xi[-1,0])
			x = xi[:,0]
			y = xi[:,1]
		
		else:
			for j, x in enumerate(xi):
				if x[1] > min([f(x[0]) for f in fs]):
					break
			fs.append(interpolate.interp1d(xi[:j,0], xi[:j,1],
											fill_value="extrapolate"))
			xm.append(xi[j-1,0])
			x = xi[:j,0]
			y = xi[:j,1]
		
		plt.plot(x, y, color=c[i%7], ls=ls)
		df = pd.DataFrame({'x': x, 'y': y},
							columns=['x', 'y'])
		df.to_csv(path+'t_%.10f.csv'%t,
					index=False)

		if xe is not None:
			plt.plot(xe[0], xe[1], 'o', color=c[i%7])

	plt.grid()
	plt.axis('equal')
	plt.show()

def read_barrier(feature=['x','y', 'L'], label='t', normalize=False):
	dd = pd.DataFrame({'x': [], 'y':[]})
	for root, dirs, files in os.walk('bdata'):
		if 'L' in root:
			par = root.split('/')
			a = float(par[1].split('_')[-1])
			r = float(par[2].split('_')[-1])
			L = float(par[3].split('_')[-1])
			n = BarrierNormlizer(TDSISDFixedPhiParam(r, a))

			for f in files:

				t = float('.'.join(f.split('_')[-1].split('.')[:-1]))
				d = pd.read_csv('%s/%s'%(root, f))
				
				# print('-------------------------------')
				# print(d['x'][10], d['y'][10], t)
				
				# normalize x, y, z to roughly within 0, 1
				if normalize:
					d['x'] = n.x_(d['x'], L)
					d['y'] = n.y_(d['y'], L, t)
					d['t'] = n.t_(t, L)
				else:
					d['t'] = t
					
				d['L'] = L

				# print(d['x'][10], d['y'][10], d['t'][10])

				# print(n.x(d['x'][10], L), 
				# 		n.y(d['y'][10], L, t), 
				# 		n.t(d['t'][10], L))

				dd = pd.concat([dd, d], ignore_index=True)

	# print(d.head)

	return dd[feature].to_numpy(), dd[label].to_numpy()


class BarrierNormlizer(object):
	"""docstring for BarrierNormlizer"""
	def __init__(self, game):

		self.g = game
		self.r = game.r
		self.a = game.a

	def x_(self, x, L):
		return x/(L/2)
	
	def x(self, x_, L):
		return x_*(L/2)

	def t_(self, t, L):
		tmin, tmax = self.g.get_trange_fromL(L)
		tstar = self.g.barrier_e_tmin(L)
		t_ = (t - tstar)/(tmax - tstar)
		return t_

	def t(self, t_, L):
		tmin, tmax = self.g.get_trange_fromL(L)
		tstar = self.g.barrier_e_tmin(L)
		t = t_*(tmax - tstar) + tstar
		return t

	def y_(self, y, L, t):
		ymax = self.a*t - sqrt((t + self.r)**2 - L**2/4)
		y_ = y/ymax
		return y_

	def y(self, y_, L, t):
		ymax = self.a*t - sqrt((t + self.r)**2 - L**2/4)
		y = y_*ymax
		return y


if __name__ == '__main__':

	# g = TDSISDFixedPhiParam(1, 1.4)
	# for L in np.linspace(10, 2.1, 5):
	# 	tmin, tmax = g.get_trange_fromL(L)
	# 	tstar = g.barrier_e_tmin(L)
	# 	print(tstar, tmax)
	read_barrier()


# plt.grid()
# plt.axis('equal')
# plt.show()

# for cnt, t in zip(cnts, ts):
# 	cv2.drawContours(drawing,[cnt],0,(int(t*100)%255,255,255),1)
# canvas[:, 1:100] = [100, 180, 250]

# cv2.imshow('output',canvas)
# cv2.waitKey(0)