import numpy as np
from math import sqrt, pi, sin, cos, tan, atan2, asin, acos
from scipy.integrate import quad, ode
from scipy.special import ellipeinc, ellipe
from pynverse import inversefunc

from util import dot, cross, norm, dist
		
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
		return 0.1*t**2 + 0.3*t

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
		if t == 0:
			return pi-self.gmm

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
		# print(t, min(t, solver.t + dt))
		while solver.successful() and solver.t < t:
			te = min(t, solver.t + dt)
			solver.integrate(te)
			ts.append(solver.t)
			ss.append(solver.y)

		return ts, ss

	def get_trange_fromL(self, L):

		tmin = 0.5*L - self.r
		tmax = 0.5*L/sin(self.gmm) - self.r
		# tbar = self.barrier_e_tmin(L)

		return tmin, tmax

	def traj_r(self, t, T=2, dt=0.05):
		# input:  t: total time
		#		  T: the moment swhich from phase 2 to phase 1
		# output: t, theta, location of the right defender,
		#			 location of the intruder


		if t == 0:
			ts = [0]
			thetas = [np.array([pi - self.gmm])]
			x2s = [np.array([self.r*sin(self.gmm), 
							self.r*cos(self.gmm)])]
			xis = [np.array([0, 0])]
			vd, vi = self.dx(t)

		# compute state
		else:
			if t <= T: 	# only Phase 2
				ts, ss = self.phase2(t, dt=dt)
				vd, vi = self.dx(t) 	# velocity (backward in time)
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

		return ts, np.asarray(thetas), \
				np.asarray(x2s), np.asarray(xis), \
				-vd, -vi

	def xd1(self, L, d, xd2):

		# find xd1 = [x, y], the intersection of:
		# C1: x^2 + y^2 = d^2
		# C2: (x - xd2)^2 + (y - yy2)^2 = L^2

		a = xd2[0]
		b = xd2[1]
		A = a**2 + b**2
		B = A + d**2 - L**2

		D = 4*A*d**2 - B**2

		# there's no solution if 
		# C1 and C2 don't intersect
		if D < 0:
			return None

		x = (a*B - b*sqrt(D))/(2*A)
		y = (B - 2*a*x)/(2*b)

		return np.array([x, y])

		
	def point_on_barrier(self, L, t, T):

		_, _, x2s, xis, vd2, vi = self.traj_r(t, T=T)
		xc = xis[0]
		xi = xis[-1]
		x2 = x2s[-1]
		x1 = self.xd1(L, t+self.r, x2)

		if x1 is None:
			# print('no solution for x1 at', t, T)
			x1 = np.array([L/2, 0])
			x2 = np.array([-L/2, 0])

			# x1, x2, 
			# xi, xc vd1, vd2, vi all None
			return x1, x2, None, None, None, None, None
		
		vd1 = -x1/norm(x1) # velocity, from x1 to [0, 0]

		vd = x2 - x1 		# vector from D1 to D2
		ed = vd/norm(vd)
		ed = np.array([ed[0], ed[1], 0])
		ex = np.array([1, 0, 0])

		sin_a = cross(ex, ed)[-1]
		cos_a = dot(ex, ed)
		C = np.array([[cos_a, sin_a],
					 [-sin_a, cos_a]])

		xm = 0.5*(x1 + x2)	# mid point of |D1 D2|

		# translate locations
		xc = C.dot(xc - xm)
		xi = C.dot(xi - xm)
		x1 = C.dot(x1 - xm)
		x2 = C.dot(x2 - xm)

		# translate velocities
		vd1 = C.dot(vd1)
		vd2 = C.dot(vd2)
		vi = C.dot(vi)

		return x1, x2, xi, xc, vd1, vd2, vi

	def barrier_e_tmin(self, L):

		def t2D(t):

			# D>0 means there exists a solution for xd1
			Dmin = 1e10

			for T in np.linspace(0, t, 10):
				_, _, x2s, xis, _, _ = self.traj_r(t, T=T)
				d = t + self.r
				xd2 = x2s[-1]
				s = norm(xd2)
				a = xd2[0]
				b = xd2[1]
				A = a**2 + b**2
				B = A + d**2 - L**2

				D = 4*A*d**2 - B**2
				# print(D)
				Dmin = min(D, Dmin)

			return Dmin


		# print(t2D(0))
		# tmin, tmax = self.get_trange_fromL(L)
		# for t in np.linspace(tmin, tmax, 20):
		# 	print(t, t2D(t))
		
		# tt = inversefunc(t2D, y_values=1e-10)		
		# print('------------------------')
		# print(tt)

		return inversefunc(t2D, y_values=1e-10)


	def barrier_e(self, L, t):
		
		res = [[] for _ in range(7)]
		for T in np.linspace(0, t, 50):
			x = self.point_on_barrier(L, t, T)

			# break when it start to have no solution for xd1
			if x[-1] is None:
				break

			for r, xx in zip(res, x):
				r.append(xx)

		for i in range(4):
			res[i] = np.asarray(res[i])

		# x1, x2, xi, xc
		return res[2], res[3],\
				res[4], res[5], res[6]			# velocities: v1, v2, vi


	def barrier_n(self, L, t): 
		# natrual barrier
		gmm = asin(0.5*L/(t + self.r))
		c = np.array([0, -(t + self.r)*cos(gmm)])
		d = self.a*t

		xis, xcs = [], []
		v1s, v2s, vis = [], [], []
		for g in np.linspace(0, gmm-self.gmm, 20):
			e = np.array([sin(g), cos(g)])
			x1 = np.array([-L/2, 0])
			x2 = np.array([ L/2, 0])
			x = c + d*e
			xis.append(x)
			xcs.append(c)

			v1 = c-x1
			v2 = c-x2
			v1s.append(v1/norm(v1))
			v2s.append(v2/norm(v2))
			vis.append(-self.a*e)

		return np.asarray(xis), np.asarray(xcs),\
				np.asarray(v1s), np.asarray(v2s),\
				np.asarray(vis)


	def barrier(self, L, t, onlye=False):

		tmin, tmax = self.get_trange_fromL(L)
		assert tmin <= t <= tmax
		xi_e, xc_e, v1_e, v2_e, vi_e = self.barrier_e(L, t)
		xi_n, xc_n, v1_n, v2_n, vi_n = self.barrier_n(L, t)
		# print(xi_e)

		if len(xi_e)>0:
			# print(v1_n)
			xi = np.concatenate((xi_n, xi_e), 0)
			xc = np.concatenate((xc_n, xc_e), 0)
			v1 = np.concatenate((v1_n, v1_e), 0)
			v2 = np.concatenate((v2_n, v2_e), 0)
			vi = np.concatenate((vi_n, vi_e), 0)
		else:
			xi = xi_n
			xc = xc_n
			v1 = v1_n
			v2 = v2_n
			vi = vi_n

		return xi, xc, v1, v2, vi
		
		

