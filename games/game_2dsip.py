import os
import numpy as np
from math import atan2, acos, sqrt, pi

from sklearn import linear_model
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
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

class TDSISDSps():
	"""docstring for TDSISDPointCapGame"""
	def __init__(self, r, vd, vi):

		self.vi = vi
		self.vd = vd
		self.a = vi/vd
		self.gmm = acos(vd/vi)
		
		# compute the semipermeable surface
		self.g = TDSISDHagdorn(r, vd, vi)

		def gmm(k):
			gmm_min = self.g.lb/2
			gmm_max = pi/2
			return gmm_min + k*(gmm_max - gmm_min)

		ax = plt.figure().add_subplot(projection='3d')
		w0, w1, w2 = self.get_barrier_data(7, self.g.lb/2, 0, ax=ax, c='r')
		# w0, w1, w2 = self.get_barrier_data(5, gmm(0.8), 0, ax=ax, c='g')
		# w0, w1, w2 = self.get_barrier_data(7, self.g.lb/2, 0)
		# w0, w1, w2 = self.get_barrier_data(5, gmm(0.8), 0)
		plt.show()	

		self.fx = w1
		self.fy = w2
		self.fL = -1

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

	def ee(self, x, y, L):
		den = sqrt(self.fx**2 + self.fy**2)
		return -np.array([self.fx, self.fy])/den	

	def pp(self, x, y, L):
		yfx_xfy = y*self.fx - x*self.fy
		p1 = np.array([self.fx/2 + self.fL, self.fy/2 + yfx_xfy/L])
		p2 = np.array([self.fx/2 - self.fL, self.fy/2 - yfx_xfy/L])
		return p1, p2

	# def update(self, x1, x2):
	# 	self.x1 = np.array([x for x in x1])
	# 	self.x2 = np.array([x for x in x2])
	# 	self.L, self.xm, self.C, self.theta = self.get_transform()

	def get_barrier_data(self, t, gmm, dlt, n=20, ax=None, c='r'):

		'''
		input: 	t: 		total time spent on the trajector
				gmm:	gmm at capture (value of the game)
				dlt:	dlt for natural barrier
		'''

		# natural barrier (all straight lines)
		x, t = self.g.tdsi_nbarrier(t, gmm, 0, frame='xyL')	
		XY = x[:,:2] # x, y
		L = x[:, 2]  # L

		if ax is not None: 
			ax.plot(x[:,0], x[:,1], x[:,2], c+'-o', markersize=2)

		for dlt in np.linspace(0, gmm-self.g.lb/2, 10):
			x, t = self.g.tdsi_nbarrier(t, gmm, dlt, frame='xyL')	
			XY = x[:,:2] # x, y
			L = x[:, 2]  # L

			if ax is not None: 
				ax.plot(x[:,0], x[:,1], x[:,2], c+'-o', markersize=2)

		# envelope barrier. Phase I: straight line trajectory, Phase II curved trajectory
		phi_min = self.g.lb 				# minimum phi possible. See Hagedorn 1976.

		# loop over different phi	
		for k in np.linspace(.01, .99, n): 
			phi_max = self.g.get_max_phi(gmm=gmm)		# maximum phi, determined by gmm. See Hagedorn 1976.
			phi = phi_min + k*(phi_max - phi_min)

			t2 = self.g.t_from_phi(phi) 	# time spent on phase II
			tau = t - t2					# time spend on phase I

			# get the envelop barrier
			if tau < 0: break;
			x, t = self.g.tdsi_ebarrier(phi, tau, gmm, n=20, frame='xyL')

			if ax is not None: ax.plot(x[:,0], x[:,1], x[:,2], c+'-o', markersize=2)

			XY = np.concatenate((XY, x[:,:2]))
			L = np.concatenate((L, x[:, 2]))

		# linear regression to fit the semipermeable surface.
		reg = linear_model.LinearRegression()
		model = reg.fit(XY, L)
		r = model.score(XY, L)

		# L = w1*X + w2*Y + w0. Or, w1*X + w2*Y - L + w0 = 0
		w0 = reg.intercept_
		w1, w2 = reg.coef_

		if ax is not None:
			# check the result, a single value
			print('-------'*10)
			print('fitting score: %.2f'%r)
			print('parameters: w0=%.2f, w1=%.2f, w2=%.2f'%(w0, w1, w2))
			print('mannual result = %.5f'%(w0 + w1*1 + w2*2),
					'predicted result = %.5f'%model.predict([[1, 2]]))

			# plot
			X = np.arange(-15, 1, 0.25)
			Y = np.arange(-.5, 5, 0.25)
			X, Y = np.meshgrid(X, Y)
			XY = np.array([X.flatten(), Y.flatten()]).T
			Z = model.predict(XY).reshape(X.shape)

			# ax.plot_surface(X, Y, Z)
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('L')

		return w0, w1, w2

class TDSISDGame():
	"""docstring for TDSISDPointCapGame"""
	def __init__(self, vd, vi, r,  
						x1, x2, xi,
						dt=.01):		

		self.sps = TDSISDSps(r, vd, vi)
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

		if norm(x1 - xi) < self.r or norm(x2 - xi) < self.r:
			return True

		return False

	def play(self, render=False, record=False):

		xs = [p.x for p in self.players]
		x1s, x2s, xis = [xs[0]], [xs[1]], [xs[2]]

		for i in range(400):

			# print('--------------------', i, self.t, '--------------------')

			v1, v2, vi = self.strategy_barrier(*xs)
			vs = [v1, v2, vi]

			# take one step
			for p, v in zip(self.players, vs):
				p.step(v)
			self.t += self.dt
			# self.sps.update(self.D1.x, self.D2.x)

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
				break

		return np.asarray(x1s), np.asarray(x2s), np.asarray(xis)


	def strategy_barrier(self, x1, x2, xi):
		
		x, y, L, xm, C, theta = self.sps.get_transform(x1, x2, xi)

		p1, p2 = self.sps.pp(x, y, L)
		ee = self.sps.ee(x, y, L)

		print(ee)

		vi =  self.vi*ee/norm(ee)
		v1 = -self.vd*p1/norm(p1)
		v2 = -self.vd*p2/norm(p2)

		Cinv = np.linalg.inv(C)

		return Cinv.dot(v1), Cinv.dot(v2), Cinv.dot(vi)


if __name__ == '__main__':
	game = TDSISDGame(1, 1.5, 1,
					np.array([-5, 0]),
					np.array([ 5, 0]),
					np.array([ 0, 2]))
	# x1, x2, xi = game.play()

	# plt.plot(x1[:,0], x1[:,1])
	# plt.plot(x2[:,0], x2[:,1])
	# plt.plot(xi[:,0], xi[:,1])

	# plt.show()

