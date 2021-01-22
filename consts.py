import numpy as np
from math import sqrt, pi, sin, cos, tan, atan2, asin, log
from scipy.integrate import quad
from scipy.special import ellipeinc, ellipe
import matplotlib.pyplot as plt

import rendering as rd
from util import dot, norm, dist

class MDSISDLineParam(object):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, nd):
		
		# inputs
		self.r = r
		self.a = a
		self.nd = nd

		# beta
		self.cos_b = a/sqrt(a**2 + 1)
		self.sin_b = 1/sqrt(a**2 + 1)
		self.beta = atan2(self.sin_b, self.cos_b)

		# gamma
		self.cos_gm = 2*a/(a**2 + 1)
		self.sin_gm = (a**2 - 1)/(a**2 + 1)
		self.tan_gm = self.sin_gm/self.cos_gm
		self.gamma = atan2(self.sin_gm, self.cos_gm)

		# h_r, delta_r, x_r
		self.h_r 	 = r*self.sin_b
		self.delta_r = r/self.cos_b
		self.x_r = self.h_r*(self.tan_gm + 1/self.tan_gm)
		
		# vectors
		self.w_m = np.array([self.sin_b, -self.cos_b]) # w-
		self.w_p = np.array([self.sin_b,  self.cos_b]) # w+
		self.e_L = np.array([self.cos_gm,  self.sin_gm])
		self.e_R = np.array([self.cos_gm, -self.sin_gm])

		# s1, s2, s3
		self.s1 = self.S(pi - self.beta)
		self.s2 = self.s1 - self.S(self.beta)
		self.s3 = self.h_r/(a*self.sin_gm)

		# responsible line segments
		self.L1 = self.s1 + \
				  r*(1 + self.cos_b) + \
				  self.s3*a*self.cos_gm
		self.L2 = self.s2 + \
				  2*r*self.cos_b + \
				  2*self.s3*a*self.cos_gm
		self.D = [0] + \
				 [self.L1 + j*self.L2 for j in range(self.nd-1)] + \
				 [2*self.L1 + (nd-1)*self.L2]
		
		# defenders' initial conditions
		self.xd = [r] + \
				  [d + self.delta_r for d in self.D[1:-1]]

		# some critical points
		self.a_r = [d - self.x_r for d in self.D[1:-1]] + [None]
		self.c_r = [d - self.x_r - 0.5*self.delta_r for d in self.D[1:-1]] + [None]

		self.a_l = [None] + [d + self.x_r for d in self.D[1:-1]]
		self.c_l = [None] + [d + self.x_r + 0.5*self.delta_r for d in self.D[1:-1]]

		self.x_b, self.y_b = self.cycloid_1(self.beta)
		# print(self.D)
		# print(self.a_l)

	def S(self, x):
		I, e = quad(lambda a: sqrt(self.a**2 - cos(a)**2) + sin(a), 0, x)
		return I*self.r/(self.a**2 - 1)

	def cycloid_1(self, b):
		k = self.a*self.r/(self.a**2 - 1)
		m = 1/self.a
		x = k*(ellipeinc(pi/2-b, m) + self.a*cos(b))
		y = self.r*sin(b)

		return x, y

	def cycloid(self, b):
		# pseudo cycloid
		if b >= self.beta:
			x, y = self.cycloid_1(b)
		else:
			k1 = sqrt(self.a**2 + 1)/(self.a**2 - 1)
			k2 = (1-self.a*tan(b))/(self.a + tan(b))
			t = k1*k2*self.r
			x = self.x_b + self.a*t*self.cos_gm
			y = self.y_b - self.a*t*self.sin_gm

		return np.array([x, y])


if __name__ == '__main__':
	g = MDSISDLineParam(1.1, 1.4, 5)
	
	p = []
	for b in np.linspace(pi/2, 0, 25):
		p.append(g.cycloid(b))
	p = np.asarray(p)

	plt.plot(p[:,0], p[:,1])
	plt.plot(g.x_b, g.y_b, 'o')
	plt.plot()
	plt.grid()
	plt.axis('equal')
	plt.show()
