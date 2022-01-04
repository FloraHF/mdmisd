import os
import numpy as np
from math import atan2, acos, sqrt, pi

from sklearn import linear_model
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
sys.path.append(".")

from util import norm, dot, cross

from bases.base_2dsihg import TDSISDHagdorn

import rendering as rd

dc = [np.array([0.1, 1., 0.05]),
		np.array([0.4, 0.05, 1.]),
		np.array([0.8, 0.05, 1.]),
		np.array([0.05, 0.5, 0.1]),
		np.array([.7, 1., .5])]
ic = np.array([0.1, 1., 1.])

class TDSISDPlayer(object):

	def __init__(self, x, vmax, dt=.1, 
					name='', 
					size=.1,
					color='b'):
		
		# inputs
		self.vmax = vmax
		self.x0 = x
		self.dt = dt

		# for rendering
		self.name = name
		self.size = size
		self.color = color

		self.reset()

	def reset(self):
		self.x = np.array([xx for xx in self.x0])
		self.v = np.array([0, 0])
		
	def step(self, v):
		self.v = np.array([vv for vv in v])
		# print(type(self.dt))
		self.x = self.x + self.v*self.dt	

class TDSISDGame():
	"""docstring for TDSISDPointCapGame"""
	def __init__(self, r, vd, vi,
						x1, x2, xi,
						dt=.1):		

		self.sps = TDSISDHagdorn(r, vd, vi)
		self.t = 0
		self.dt = dt

		self.x1 = x1
		self.x2 = x2

		self.vi = vi
		self.vd = vd
		self.a = vi/vd
		self.gmm = acos(vd/vi)
		self.r = r

		# players
		self.D1 = TDSISDPlayer(x1, vd, dt=dt, name='D1', color=dc[0])
		self.D2 = TDSISDPlayer(x2, vd, dt=dt, name='D2', color=dc[1])
		self.I  = TDSISDPlayer(xi, vd, dt=dt, name='I', color=ic)

		self.players = [self.D1, self.D2, self.I]		

	def iscap(self, x1, x2, xi):

		k = 0.8
		if norm(x1 - xi) < k*self.r or norm(x2 - xi) < k*self.r:
			return True

		return False

	def get_transform(self, x1, x2, xi):

		# translation
		xm = 0.5*(x1 + x2)

		# L
		vd = x2 - x1
		L = norm(vd)

		# rotation
		ed = vd/L
		ed = np.array([ed[0], ed[1], 0])
		ex = np.array([1, 0, 0])

		sin_a = cross(ex, ed)[-1]
		cos_a = dot(ex, ed)
		C = np.array([[cos_a, sin_a],
					 [-sin_a, cos_a]]) # to F_D1D2

		xi = C.dot(xi - xm)
		x = xi[0]
		y = xi[1]

		return x, y, L, xm, C, atan2(sin_a, cos_a)

	def strategy_barrier(self, x1, x2, xi):
		
		x, y, L, xm, C, theta = self.get_transform(x1, x2, xi)

		gmm = self.sps.solve_gmm(L, x, y)
		fL, fg, fx, fy = self.sps.SPS_df(x, y, L, gmm)

		p1 = np.array([fx/2 + fL, fy/2 + (y*fx - x*fy)/L])
		p2 = np.array([fx/2 - fL, fy/2 - (y*fx - x*fy)/L])
		pe = np.array([fx, fy])
		p1 = p1/norm(p1)
		p2 = p2/norm(p2)
		p2 = p2/norm(p2)

		vi =  self.vi*pe/norm(pe)
		v1 =  self.vd*p1/norm(p1)
		v2 =  self.vd*p2/norm(p2)

		Cinv = np.linalg.inv(C)

		return Cinv.dot(v1), Cinv.dot(v2), Cinv.dot(vi)

	def play(self, render=False, record=False):

		xs = [p.x for p in self.players]
		x1s, x2s, xis = [xs[0]], [xs[1]], [xs[2]]


		while self.t < 5:

			# print('--------------------', i, self.t, '--------------------')
			v1, v2, vi = self.strategy_barrier(*xs)
			vs = [v1, v2, vi]

			# take one step
			for p, v in zip(self.players, vs):
				p.step(v)
			self.t += self.dt

			# record data
			xs = [p.x for p in self.players]
			x1s.append(xs[0])
			x2s.append(xs[1])
			xis.append(xs[2])

			# rendering
			if render:
				self.render()

			# break upon capture
			if self.iscap(*xs):
				print('captured')
				break
				
		return np.asarray(x1s), np.asarray(x2s), np.asarray(xis)

if __name__ == '__main__':
	game = TDSISDGame(1, 1, 1.5,
						np.array([-3.5, 0]),
						np.array([ 3.5, 0]),
						np.array([-2.4, 1.25]))
	x1, x2, xi = game.play()

	plt.plot(x1[:,0], x1[:,1], '-o')
	plt.plot(x2[:,0], x2[:,1], '-o')
	plt.plot(xi[:,0], xi[:,1], '-x')

	plt.show()

