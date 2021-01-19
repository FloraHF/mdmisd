import numpy as np
from math import sqrt, pi, sin, cos, atan2

from scipy.integrate import quad

from util import dot, norm, dist


class MDSISDLineDefender(object):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, j, nd):
		
		# inputs
		self.r = r
		self.a = a
		self.j = j
		self.nd = nd

		# beta
		self.cos_b = a/sqrt(a**2 + 1)
		self.sin_b = 1/sqrt(a**2 + 1)
		self.beta = atan2(self.sin_b, self.cos_b)

		# gamma
		self.cos_gm = 2*a/sqrt(a**2 + 1)
		self.sin_gm = (a**2 - 1)/(a**2 + 1)
		self.tan_gm = self.sin_gm/self.cos_gm

		# h_r, delta_r, x_r
		self.h_r 	 = r*self.sin_b
		self.delta_r = r/self.cos_b
		self.x_r = self.h_r*(self.tan_gm + 1/self.tan_gm)
		
		# vectors
		self.w_l = np.array([self.sin_b, -self.cos_b]) # w-
		self.w_r = np.array([self.sin_b,  self.cos_b]) # w+
		self.e_L = np.array([self.cos_gm,  self.sin_gm])
		self.e_R = np.array([self.cos_gm, -self.sin_gm])

		# s1, s2, s3
		self.s1 = self.S(pi - self.beta)
		self.s2 = self.s1 - self.S(self.beta)
		self.s3 = self.h_r/(a*self.sin_gm)

		# responsible line segment
		self.L1 = self.s1 + \
				  r*(1 + self.cos_b) + \
				  self.s3*a*self.cos_gm
		self.L2 = self.s2 + \
				  2*r*self.cos_b + \
				  2*self.s3*a*self.cos_gm

		self.D_r = self.get_D(j)
		self.D_l = self.get_D(j-1)

		# initial location
		self.x0 = self.get_x0(j)

		# critical points on the line
		self.a_r, self.c_r = self.get_right(j)
		self.a_l, self.c_l = self.get_left(j)

		# state
		self.x = self.x0
		self.v = np.array([0, 0])

	def get_D(self, j):
		if j < 0:
			D = 0
		elif j == self.nd-1:
			D = 2*self.L1 + (j-1)*self.L2
		else:
			D = self.L1 + j*self.L2
		return D

	def get_x0(self, j):
		if j == 0:
			x0 = self.D_l + self.r
		else:
			x0 = self.D_l + self.delta_r
		return np.array([x0, 0])

	def get_right(self, j):
		if j = self.nd - 1:
			a = None
			c = None
		else:
			a = self.D_r - self.x_r
			c = self.D_r - self.x_r - 0.5*self.delta_r
		return a, c

	def get_left(self, j):
		if j = 0:
			a = None
			c = None
		else:
			a = self.D_l + self.x_r
			c = self.D_l + self.x_r + 0.5*self.delta_r
		return a, c

	def in_CR(self, y):
		return self.a_r is not None and abs(y[1]) < (y[0] - self.a_r)/self.tan_gm

	def in_CL(self, y):
		return self.a_l is not None and abs(y[1]) < (self.a_l - y[0])/self.tan_gm

	def strategy_x(self, y):
		# defending strategy based on location only
		if self.D_l <= y[0] <= self.D_r:
			if self.in_CR(y):
				pass
			elif self.in_CL(y):
				pass
			else:
				vx = 1
		else:
			vx = 0.
		return np.array([vx, 0])

	def strategy_v(self, y, vi):
		# defending strategy based on location and velocity
		if self.D_l <= y[0] <= self.D_r:
			if self.in_CR(y):
				vx = dot(self.e_R, vi)/self.a
			elif self.in_CL(y):
				vx = dot(self.e_L, vi)/self.a
			else:
				vx = vi[0]
		else:
			vx = 0.

		return np.array([vx, 0])

class MDSISDLineIntruder(object):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, j, nd):
		
		# inputs
		self.r = r
		self.a = a
		self.j = j
		self.nd = nd

		# state
		self.x = np.array([0, 0])
		self.v = np.array([0, 0])

		# beta
		self.cos_b = a/sqrt(a**2 + 1)
		self.sin_b = 1/sqrt(a**2 + 1)
		self.beta = atan2(self.sin_b, self.cos_b)

		# gamma
		self.cos_gm = 2*a/sqrt(a**2 + 1)
		self.sin_gm = (a**2 - 1)/(a**2 + 1)
		self.tan_gm = self.sin_gm/self.cos_gm

		# h_r, delta_r, x_r
		self.h_r 	 = r*self.sin_b
		self.delta_r = r/self.cos_b
		self.x_r = self.h_r*(self.tan_gm + 1/self.tan_gm)
		
		# vectors
		self.w_l = np.array([self.sin_b, -self.cos_b]) # w-
		self.w_r = np.array([self.sin_b,  self.cos_b])


	def get_w(self, z):
		# see Fig.1 in the paper
		e = z/norm(z)
		w = np.array([e[1], -e[0]])
		Rw = np.array([-e[0], -e[1]])
		return w, Rw

		
	def get_effectxd(self, xds):
		# return the x-coordinate of the effective defender
		JL, JR = [], []
		jL, jR = [], []
		for j, xd in enumerate(xds):
			if xd[0] <= self.x[0]:
				JL.append(xd[0])
				jL.append(j)
			else:
				LR.append(xd[0])
				jR.append(j)

		if JL:
			x_l = max(JL)
			jxl = jL[JL.index(x_l)]
		else:
			x_l = min(JR)
			jxl = jR[JR.index(x_l)]

		if JR:
			x_r = min(JR)
			jxr = jR[JR.index(x_r)]
		else:
			x_r = max(JL)
			jxr = jL[JL.index(x_r)]

		x_m = 0.5*(x_l + x_r)
		if self.x[0] < x_m:
			ksi = x_l
			je = jxl
		else:
			ksi = x_r
			je = jxr

		return np.array([ksi, 0]), je


	def strategy(self, xds, vd):
		xd, jd = self.get_effectxd(xds)
		w, Rw = self.get_w(self.x - xd)
		vd_w = dot(vd, w)
		vd_Rw = dot(vd, Rw)

		vi_Rw = vd_Rw
		vi_w = sqrt(a**2 - vi_Rw**2)

		return = vi_Rw*Rw + vi_w*w

class MDSISDLineGame(object):
	""" Parameters for multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, nd):
		# input a: speed ratio vi/vd
		# input r: capture range of the defender

		# inputs:
		self.r = r
		self.a = a
		self.nd = nd

		# beta
		self.cos_b = a/sqrt(a**2 + 1)
		self.sin_b = 1/sqrt(a**2 + 1)
		self.beta = atan2(self.sin_b, self.cos_b)

		# gamma
		self.cos_gm = 2*a/sqrt(a**2 + 1)
		self.sin_gm = (a**2 - 1)/(a**2 + 1)
		self.tan_gm = self.sin_gm/self.cos_gm

		# h_r, delta_r, x_r
		self.h_r 	 = r*self.sin_b
		self.delta_r = r/self.cos_b
		self.x_r = self.h_r*(self.tan_gm + 1/self.tan_gm)
		
		# vectors
		self.w_l = np.array([self.sin_b, -self.cos_b]) # w-
		self.w_r = np.array([self.sin_b,  self.cos_b])

		# s1, s2, s3
		self.s1 = self.S(pi - self.beta)
		self.s2 = self.s1 - self.S(self.beta)
		self.s3 = self.h_r/(a*self.sin_gm)

		# Delta
		self.L1 = self.s1 + \
				  r*(1 + self.cos_b) + \
				  self.s3*a*self.cos_gm
		self.L2 = self.s2 + \
				  2*r*self.cos_b + \
				  2*self.s3*a*self.cos_gm

		self.D = [self.L1] + \
				 [self.L1 + k*self.L2 for k in range(1,nd-1)] + \
				 [2*self.L1 + (nd - 1)*self.L2]

		self.xd = [r] + \
				  [self.delta_r + d for d in self.D[1:]]

		self.a_r = [d - self.x_r for d in self.D[:-1]] + [None]
		self.c_r = [d - self.x_r - 0.5*self.delta_r for d in self.D[:-1]] + [None]

		self.a_l = [None] + [d + self.x_r for d in self.D[1:]]
		self.c_l = [None] + [d + self.x_r + 0.5*self.delta_r for d in self.D[1:]]

	def S(self, x):
		I, e = quad(lambda a: sqrt(self.a**2 - cos(a)**2) + sin(a), 0, x)
		return I*self.r/(self.a**2 - 1)


if __name__ == '__lain__':
	g = MDSISDLineGame(1, 1.2, 5)
	print(g.xd)
	print(g.a_r)
	print(g.a_l)