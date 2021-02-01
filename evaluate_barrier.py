import os

import numpy as np
from math import pi, cos, sin, log, sqrt
from scipy import interpolate
import pandas as pd

import matplotlib.pyplot as plt

import cv2

from util import norm
from base_2dsi import TDSISDFixedPhiParam

c = ['k', 'b', 'r', 'm', 'y', 'g', 'c']

def plot_barrier(L, ls='-'):
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
		# plt.plot(xc[:,0], xc[:,1], color=c[i%7])

		if xe is not None:
			plt.plot(xe[0], xe[1], 'o', color=c[i%7])

	plt.grid()
	plt.axis('equal')
	plt.show()

def read_bdata(path):
	dd = pd.DataFrame({'x': [], 'y':[]})
	for root, dirs, files in os.walk('bdata'):
		if 'L' in root:
			par = root.split('/')
			a = float(par[1].split('_')[-1])
			r = float(par[2].split('_')[-1])
			L = float(par[3].split('_')[-1])
			for f in files:
				t = float('.'.join(f.split('_')[-1].split('.')[:-1]))
				d = pd.read_csv('%s/%s'%(root, f))
				d['t'] = t
				d['L'] = L

				dd = pd.concat([dd, d], ignore_index=True)

	# k1, k2 = 1210, 1250
	# print(dd['t'][k1:k2])
	# plt.plot(dd['x'][k1:k2], dd['y'][k1:k2])
			
	# plt.show()

	return dd[['x','y', 'L']].to_numpy(), dd['t'].to_numpy()

	# return dd['']
			



if __name__ == '__main__':

	# plot_barrier(10)
	# plot_barrier(9)
	# plot_barrier(8)
	# plot_barrier(7)
	# plot_barrier(6)
	# plot_barrier(5)
	# plot_barrier(4)
	# plot_barrier(3)
	# plot_barrier(2.0001)

	xx, yy = read_bdata('')

	k, s = 0, 0
	x_ = []
	y_ = yy[0]
	L_ = xx[0,2]
	for x, y in zip(xx, yy):
		if y == y_:
			x_.append(x)
		else:
			x__ = np.asarray(x_)
			plt.plot(x__[:,0], x__[:,1], color=c[s%7])
			x_ = []
			y_ = y
			k += 1
		if x[2] != L_:
			L_ = x[2]
			s += 1
	plt.show()





# plt.grid()
# plt.axis('equal')
# plt.show()

# for cnt, t in zip(cnts, ts):
# 	cv2.drawContours(drawing,[cnt],0,(int(t*100)%255,255,255),1)
# canvas[:, 1:100] = [100, 180, 250]

# cv2.imshow('output',canvas)
# cv2.waitKey(0)