import numpy as np
import matplotlib.pyplot as plt

from util import norm
from base_2dsir0 import TDSISDPointCap

class TDSISDPointCapPlayer(object):

	def __init__(self, x, vmax, dt=.1, name='', color='b'):
		
		# inputs
		self.vmax = vmax
		self.x0 = x
		self.dt = dt

		# for rendering
		self.color = color

		self.reset()

	def reset(self):
		self.x = np.array([xx for xx in self.x0])
		self.v = np.array([0, 0])
		
	def step(self, v):
		self.v = np.array([vv for vv in v])
		# print(type(self.dt))
		self.x = self.x + self.v*self.dt	


class TDSISDPointCapGame(TDSISDPointCap):
	"""docstring for TDSISDPointCapGame"""
	def __init__(self, vd, vi, dt=.2):		
		super(TDSISDPointCapGame, self).__init__(vd, vi)

		self.t = 0.
		self.dt = dt
		L, x = 10, 2
		self.D1 = TDSISDPointCapPlayer(np.array([-L/2, 0.]), 
										vd, dt=dt, name='D1', color='b')
		self.D2 = TDSISDPointCapPlayer(np.array([L/2, 0.]), 
										vd, dt=dt, name='D2', color='b')
		ymin, ymax = self.isc.yrange(L, x)
		self.I  = TDSISDPointCapPlayer(np.array([x, .5*(ymin+ymax)]), 
										vd, dt=dt, name='I', color='r')
		self.players = [self.D1, self.D2, self.I]

		self.strategy = {'b':self.strategy_barrier,
						 'p':self.strategy_pass,
						 'd':self.strategy_default,
						 't':self.strategy_tdefense}

	def iscap(self, x1, x2, xi):

		if norm(x1 - xi)<.1 or norm(x2 - xi)<.1:
			return True

		return False

	def play(self, dstr='b', istr='p'):
		xs = [p.x for p in self.players]
		x1s, x2s, xis = [xs[0]], [xs[1]], [xs[2]]

		for i in range(50):

			if dstr == 't':
				v1, v2 = self.strategy[dstr](*xs, self.I.v)
			else:
				v1, v2, _ = self.strategy[dstr](*xs)
			_, _, vi  = self.strategy[istr](*xs)
			vs = [v1, v2, vi]

			for p, v in zip(self.players, vs):
				p.step(v)
			self.t += self.dt

			xs = [p.x for p in self.players]
			x1s.append(xs[0])
			x2s.append(xs[1])
			xis.append(xs[2])

			if self.iscap(*xs):
				break

		print('%s(D) v.s %s(I), x_c=[%.2f, %.2f], t=%.2f'%
			  (dstr, istr, self.I.x[0], self.I.x[1], self.t))

		return np.asarray(x1s), np.asarray(x2s), np.asarray(xis)

	def reset(self):
		for p in self.players:
			p.reset()
		self.t = 0.

if __name__ == '__main__':
	g = TDSISDPointCapGame(1, 1.4)
	x1, x2, xi = g.play(dstr='b', istr='b')
	plt.plot(x1[:,0], x1[:,1])
	plt.plot(x2[:,0], x2[:,1])
	plt.plot(xi[:,0], xi[:,1])
	plt.grid()
	plt.axis('equal')
	plt.show()