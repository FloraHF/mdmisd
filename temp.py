import numpy as np
from math import sqrt, pi, sin, cos, atan2

from scipy.integrate import quad

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
		self.cos_gm = 2*a/sqrt(a**2 + 1)
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
		# print(self.D)
		# print(self.a_l)

	def S(self, x):
		I, e = quad(lambda a: sqrt(self.a**2 - cos(a)**2) + sin(a), 0, x)
		return I*self.r/(self.a**2 - 1)


class MDSISDLineDefender(object):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, j, nd, dt=0.1):
		
		# inputs
		self.param = MDSISDLineParam(r, a, nd)
		self.id = j
		self.dt = dt

		# segment responsible [D_l, D_r]
		self.D_l = self.param.D[j]
		self.D_r = self.param.D[j+1]

		# initial location
		self.x0 = self.param.xd[j]

		# critical points on the line
		self.a_r = self.param.a_r[j]
		self.c_r = self.param.c_r[j]
		self.a_l = self.param.a_l[j] 
		self.c_l = self.param.c_l[j]

		# state
		self.reset()
		
	def reset(self):
		self.x = np.array([self.x0, 0])
		self.v = np.array([0, 0])

	def step(self, v):
		self.v = np.array([vv for vv in v])
		self.x = self.x + self.v*self.dt

	def in_CR(self, xi):
		return self.a_r is not None and abs(xi[1]) < (xi[0] - self.a_r)/self.param.tan_gm

	def in_CL(self, xi):
		return self.a_l is not None and abs(xi[1]) < (self.a_l - xi[0])/self.param.tan_gm

	def in_D(self, xi):
		return self.D_l <= xi[0] <= self.D_r

	def strategy_x(self, xi):
		# defending strategy based on location only

		if self.in_D: # if the intruder is in the responsible segment
			if self.in_CR(xi):
				pass
			elif self.in_CL(xi):
				pass
			else:
				vx = 1
		else:
			# return to the initial point, P control with clip
			vx = self.x0 - self.x
			if vx > 1: vx = 1
			if vx < -1: vx = -1
		return np.array([vx, 0])

	def strategy_v(self, xi, vi):
		# defending strategy based on location and velocity

		if self.in_D(xi):
			# print('defender ', self.id, 'in action')
			if self.in_CR(xi):
				# print('defender: in CR', self.id)
				# print(xi[1], xi[0], self.a_r, (xi[0] - self.a_r)/self.param.tan_gm)
				vx = dot(self.param.e_R, vi)/self.param.a
			elif self.in_CL(xi):
				print('defender: in CL', self.id)
				vx = dot(self.param.e_L, vi)/self.param.a
			else:
				vx = 1.
		else:
			vx = 0.

		return np.array([vx, 0])

class MDSISDLineIntruder(object):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, nd, dt=0.1):
		
		# inputs
		self.param = MDSISDLineParam(r, a, nd)
		self.dt = dt

		# state
		self.reset()

	def reset(self):
		# from the left end of the line
		self.x = np.array([0, 0])
		self.v = np.array([0, 0])

	def step(self, v):
		self.v = np.array([vv for vv in v])
		# print(self.v)
		# print('before', self.x)
		self.x = self.x + self.v*self.dt
		# print('after', self.x)

	def get_w(self, z, j):
		# see Fig.1 in the paper
		e = z/norm(z)
		t = atan2(e[1], e[0])
		# print(t, '!!!!!!')

		if j == 0:
			if 0 <= t < self.param.beta:
				w = self.param.w_m
			else:
				w = np.array([e[1], -e[0]])

		elif j == self.param.nd-1:
			if pi - self.param.beta < t <= pi:
				w = self.param.w_p
			else:
				w = np.array([e[1], -e[0]])

		else:
			if 0 <= t < self.param.beta:
				w = self.param.w_m
			elif pi - self.param.beta < t <= pi:
				w = self.param.w_p
			else:
				w = np.array([e[1], -e[0]])
		
		Rw = np.array([w[1], -w[0]])
		# print(w, Rw)

		return w/norm(w), Rw/norm(Rw)

	def get_locald(self, xds):
		# return the x-coordinate of the effective defender
		JL, JR = [], []
		jL, jR = [], []
		for j, xd in enumerate(xds):
			if xd[0] <= self.x[0]:
				JL.append(xd[0])
				jL.append(j)
			else:
				JR.append(xd[0])
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


	def strategy(self, xds):
		xd, jd = self.get_locald(xds)
		w, Rw = self.get_w(self.x - xd, jd)

		vd = np.array([1, 0])
		vd_w = dot(vd, w)
		vd_Rw = dot(vd, Rw)
		# print('vd: ', vd_Rw, vd_w)

		vi_Rw = vd_Rw
		vi_w = sqrt(self.param.a**2 - vi_Rw**2)

		vi = vi_Rw*Rw + vi_w*w
		# print('vi: ', vi_w, w, vi_Rw, Rw)
		print('xi: ', self.x)

		return vi

class MDSISDLineGame(object):
	""" Parameters for multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, nd):
		# input a: speed ratio vi/vd
		# input r: capture range of the defender

		# inputs:
		self.param = MDSISDLineParam(r, a, nd)
		self.intruder = MDSISDLineIntruder(r, a, nd)
		self.defenders = [MDSISDLineDefender(r, a, j, nd) for j in range(nd)]

	def play(self):
		xds = [d.x for d in self.defenders]
		vi = self.intruder.strategy(xds)
		self.intruder.step(vi)
		for d in self.defenders:
			vd = d.strategy_v(self.intruder.x, self.intruder.v)
			d.step(vd)

		# print(vi)
		# print(self.intruder.x)

if __name__ == '__main__':
	g = MDSISDLineGame(1, 1.5, 5)
	# print(g.D)
	for i in range(35):
		g.play()
