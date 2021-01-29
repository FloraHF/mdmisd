import numpy as np
from math import sqrt, pi, sin, cos, tan, atan2, asin, acos
from scipy.integrate import quad, ode
from scipy.special import ellipeinc, ellipe
from pynverse import inversefunc

from util import dot, cross, norm, dist

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
		self.w_r = self.h_r/self.sin_gm
		
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
		self.Ds = [0] + \
				 [self.L1 + j*self.L2 for j in range(self.nd-1)] + \
				 [2*self.L1 + (nd-1)*self.L2]
		self.D = self.L2/2
		
		# defenders' initial conditions
		self.xd = [r] + \
				  [d + self.delta_r for d in self.Ds[1:-1]]

		# some critical points
		self.a_r = [d - self.x_r for d in self.Ds[1:-1]] + [None]
		self.c_r = [d - self.x_r - 0.5*self.delta_r for d in self.Ds[1:-1]] + [None]

		self.a_l = [None] + [d + self.x_r for d in self.Ds[1:-1]]
		self.c_l = [None] + [d + self.x_r + 0.5*self.delta_r for d in self.Ds[1:-1]]

		self.xi_b, self.yi_b = self.envelope(self.beta)
		self.xd_b = self.dist_d(self.beta)


	def S(self, x):
		I, e = quad(lambda a: sqrt(self.a**2 - cos(a)**2) + sin(a), 0, x)
		return I*self.r/(self.a**2 - 1)

	def t1(self, b):
		k = self.r/(self.a**2 - 1)
		m = 1/self.a**2
		t = k*(self.a*ellipeinc(pi/2-b, m) + cos(b))
		# t = ellipeinc(pi/2-b, m)
		return t

	def t1_inv(self, t):
		return inversefunc(self.t1, y_values=t)

	def dist_d(self, b):
		return self.t1(b)

	def dist_i(self, b):
		return self.a*self.t1(b)

	def dist_w(self, b):
		k1 = sqrt(self.a**2 + 1)/(self.a**2 - 1)
		k2 = (1-self.a*tan(b))/(self.a + tan(b))
		d = k1*k2*self.r*self.a
		return d

	def slope(self, b):
		num = (self.a**2 - 1)*cos(b)
		den = sqrt(self.a**2 - cos(b)**2) + self.a**2*sin(b)
		return num/den

	def cos_d(self, b):
		num = sqrt(self.a**2 - cos(b)**2) + self.a**2*sin(b)
		den = self.a*(sqrt(self.a**2 - cos(b)**2) + sin(b))
		return num/den
	
	def sin_d(self, b):
		num = (self.a**2 - 1)*cos(b)
		den = self.a*(sqrt(self.a**2 - cos(b)**2) + sin(b))
		return num/den

	def envelope_x(self, b):
		assert b >= self.beta
		k = self.a*self.r/(self.a**2 - 1)
		m = 1/self.a**2
		x = k*(ellipeinc(pi/2-b, m) + self.a*cos(b))
		return x

	def envelope_y(self, b):
		# assert b >= self.beta
		return self.r*sin(b)

	def envelope(self, b):
		# envelope: |DI| = r
		if b >= self.beta:
			x = self.envelope_x(b)
			y = self.envelope_y(b)
		else:
			d = self.dist_w(b)
			x = self.xi_b + d*self.cos_gm
			y = self.yi_b - d*self.sin_gm

		return np.array([x, y])

	def involute_x(self, b, bb):
		# identical to the x part in self.involute()
		# but not used there to avoid calling self.envelope() twice
		x = self.envelope_x(bb)
		dist = self.dist_i(bb) - self.dist_i(b)
		return x - dist*self.cos_d(bb)

	def involute_y(self, b, bb):
		# identical to the y part in self.involute()
		# but not used there to avoid calling self.envelope() twice
		y = self.envelope_y(bb)
		dist = self.dist_i(bb) - self.dist_i(b)
		return y + dist*self.sin_d(bb)

	def involute(self, b, bb, gm=0):
		# assert bb <= b
		if bb >= self.beta:
			x, y = self.envelope(bb)
			dist = self.dist_i(bb) - self.dist_i(b)
			x = x - dist*self.cos_d(bb)
			y = y + dist*self.sin_d(bb)
		else:
			assert self.gamma < gm <= pi/2
			d1 = self.dist_i(self.beta) - self.dist_i(b) # curved part
			d2 = self.w_r
			d = d1 + d2
			x = self.D - d*cos(gm)
			y = d*sin(gm)
		return np.array([x, y])

	def tmin(self, y):
		return y/self.a

	def tmax(self):
		return self.D - self.delta_r

	def ymax(self): 
		# maximum y to be on barrier
		return self.a*(self.D - self.delta_r)

	def partition(self, y, t):
		assert y >= 0
		tn = self.tmin(y) 
		tm = self.tmax()
		ts = y/(self.a*self.sin_gm) # t < (>=) ts: natural (envelope) barrier
		ym = self.a*tm 				# maximum y to be on barrier
 
		part = 'envelope'

		# in the defender's winning region
		if y > ym or (self.r < y <= ym and t > tm):
			part = 'dwin'

		# t is too small for the intruder to reach the line
		if t < tn: 
			part = 'small_t'

		# on the natural barrier, xd to be solved with xd_natural
		if y <= ym and \
			y/self.a <= t <= min(tm, ts): # include t = min(tm, ts) for the ease of calculation	
			part = 'natural'

		# # on the envelope barrier, xd to be solved with xd_envelope
		# # don't have to use self.t1 which includes elliptic integral
		# if self.r < y < ym*self.sin_gm and \
		# 	ts < t <= tmax:
		# 	part = 'envelope'

		# if self.h_r <= y <= self.r and \
		# 	ts < t <= tm - self.t1(arcsin(y/self.r)):
		# 	part = 'envelope'

		return part

	def xd_natural(self, y, t):
		xi = self.D - sqrt((self.a*t)**2 - y**2)
		xd_l = self.D - self.delta_r - t
		xd_r = self.D + self.delta_r + t
		dl = xd_l - xi
		dr = xd_r - xi
		return dl, dr

	def xd_envelope(self, y, t):
		b = self.t1_inv(self.D - self.delta_r - t)
		bb = inversefunc(lambda bb: self.involute_y(b, bb), 
						y_values=y)
		xi = self.involute_x(b, bb)

		xd_l = self.D - self.delta_r - t
		xd_r = self.D + self.delta_r + t
		dl = xd_l - xi
		dr = xd_r - xi
		return dl, dr

	def t_natural(self, y, x):
		t = sqrt(y**2 + x**2)/self.a
		return t

	def t_envelope(self, y, x):
		def t2xi(t):
			b = self.t1_inv(self.D - self.delta_r - t)
			bb = inversefunc(lambda bb: self.involute_y(b, bb), 
							y_values=y)
			xi = self.involute_x(b, bb)
			xd_l = self.D - self.delta_r - t
			dl = xd_l - xi
			return dl
		t = inversefunc(t2xi, y_values=x)
		return t
		
class TDSISDFixedPhiParam(object):
	"""Two-defender single-intruder game
	with fixed Phase 2 defending strategy"""
	def __init__(self, r, a, phi=None):
		
		self.r = r
		self.a = a
		self.gmm = acos(1/a)

		self.phi = phi if phi is not None else self.phi_default

	def phi_default(self, t):
		# t is backward
		return 0.001*t + 0.3*t

	def psi(self, t):
		return acos(cos(self.phi(t))/self.a)

	def get_theta(self, xd, xi):
		# the angle between DI and the vertical axis
		if len(xd) == 2:
			xd = np.array([xd[0], xd[1], 0])
		if len(xi) == 2:
			xi = np.array([xi[0], xi[1], 0])

		DI = xi - xd
		e = DI/norm(DI)
		ey = np.array([0, 1, 0])

		cos_t = dot(e, ey)
		sin_t = cross(ey, e)[-1]

		theta = atan2(sin_t, cos_t)

	def dtheta(self, t):
		phi = self.phi(t)
		return -(sqrt(self.a**2 - cos(phi)**2) - sin(phi))/self.r
		
	def theta(self, t):
		solver = ode(self.dtheta).set_integrator("dopri5")
		solver.set_initial_value(pi-self.gmm, 0)
		solver.integrate(t)
		return solver.y[0]

	def dx(self, t):
		# velocity at t, 
		# for phase 1 trajectory computation
		phi = self.phi(t)
		tht = self.theta(t)
		psi = acos(cos(phi)/self.a)
		dxd =  sin(tht + phi)
		dyd = -cos(tht + phi)
		dxi =  self.a*sin(tht + psi)
		dyi = -self.a*cos(tht + psi)

		return np.array([dxd, dyd]), \
				np.array([dxi, dyi])

	def dstate(self, t, s):
		tht = s[0]
		phi = self.phi(t)
		psi = acos(cos(phi)/self.a)
		dtht = -(sqrt(self.a**2 - cos(phi)**2) - sin(phi))/self.r
		dxd =  sin(tht + phi)
		dyd = -cos(tht + phi)
		dxi =  self.a*sin(tht + psi)
		dyi = -self.a*cos(tht + psi)

		return np.array([dtht, dxd, dyd, dxi, dyi])

	def phase2(self, t, dt=0.1):
		solver = ode(self.dstate).set_integrator("dopri5")
		s0 = np.array([pi - self.gmm, 			# theta
						self.r*sin(self.gmm), 
						self.r*cos(self.gmm), 	# xd0
						0, 0])					# xi0
		solver.set_initial_value(s0, 0)		# t0 = 0 (backwards)
		ts, ss = [0], [s0]
		while solver.successful() and solver.t < t:
			te = min(t, solver.t + dt)
			solver.integrate(te)
			ts.append(solver.t)
			ss.append(solver.y)

		return ts, ss

	def traj_r(self, t, T=2, dt=0.1):
		# input:  t: total time
		#		  T: the moment swhich from phase 2 to phase 1
		# output: t, theta, location of the right defender,
		#			 location of the intruder, 
		if t <= T: 	# only Phase 2
			ts, ss = self.phase2(t, dt=dt)
		else:		# Phase 1 and 2
			ts, ss = self.phase2(T, dt=dt)
			vd, vi = self.dx(ts[-1])
			xd = ss[-1][1:3] + vd*(t - ts[-1])
			xi = ss[-1][3:]  + vi*(t - ts[-1])
			ts.append(t)
			ss.append(np.concatenate(
				([self.get_theta(xd, xi)], xd, xi))
			)

		thetas, x2s, xis = [], [], []
		for s in ss:
			thetas.append(s[0])
			x2s.append(s[1:3])
			xis.append(s[3:])

		return ts, np.asarray(thetas), np.asarray(x2s), np.asarray(xis)

	def xd1(self, L, d, xd2):

		# find xd1 = [x, y] such that:
		# x^2 + y^2 = d^2
		# (x - xd2)^2 + (y - yy2)^2 = L^2

		a = xd2[0]
		b = xd2[1]
		A = a**2 + b**2
		B = A + d**2 - L**2
		x = (a*B - b*sqrt(4*A*d**2 - B**2))/(2*A)
		y = (B - 2*a*x)/(2*b)

		return np.array([x, y])

		
	def point_on_barrier(self, L, t, T):

		_, _, x2s, xis = self.traj_r(t, T=T)
		xc = xis[0]
		xi = xis[-1]
		x2 = x2s[-1]
		x1 = self.xd1(L, t+self.r, x2)
		
		vd = x2 - x1 		# vector from D1 to D2
		ed = vd/norm(vd)
		ed = np.array([ed[0], ed[1], 0])
		ex = np.array([1, 0, 0])

		sin_a = cross(ex, ed)[-1]
		cos_a = dot(ex, ed)
		C = np.array([[cos_a, sin_a],
					 [-sin_a, cos_a]])

		xm = 0.5*(x1 + x2)	# mid point of |D1 D2|

		# print(xc, xi, x1, x2)

		xc = C.dot(xc - xm)
		xi = C.dot(xi - xm)
		x1 = C.dot(x1 - xm)
		x2 = C.dot(x2 - xm)

		return x1, x2, xi, xc

	def barrier_t(self, L, t):
		
		res = [[] for _ in range(4)]
		for T in np.linspace(0.01, t, 30):
			x = self.point_on_barrier(L, t, T)
			for r, xx in zip(res, x):
				r.append(xx)

		for i in range(4):
			res[i] = np.asarray(res[i])

		# x1, x2, xi, xc
		return res[0], res[1], res[2], res[3]


	def barrier_n(self, L, t): 
		# natrual barrier
		gmm = asin(0.5*L/(t + self.r))
		c = np.array([0, -(t + self.r)*cos(gmm)])
		r = self.a*t

		xis = []
		for g in np.linspace(gmm-self.gmm, 0, 20):
			x = c + r*np.array([sin(g), cos(g)])
			xis.append(x)

		return np.asarray(xis)





