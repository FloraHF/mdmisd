import numpy as np
from math import pi, sin, cos, tan, acos, asin, atan2, sqrt, exp, log
from mpmath import ellipe, ellipf
from scipy.optimize import minimize_scalar
from sklearn import linear_model

import matplotlib.pyplot as plt

import sys
sys.path.append(".")

import os

from util import norm, dot, dot_xA, cross, plot_cap_range, normalize
# from . import 
# import os

class TDSISDHagdorn(object):
	"""docstring for TDSISDHagdorn"""
	def __init__(self, r, vd, vi):
		
		self.vd = vd
		self.vi = vi
		self.w = vi/vd
		self.l = r

		self.lb = 2*acos(1/self.w)
		self.ct = self._t_from_phi(self.lb)

		self.phi_max = self.get_max_phi()
		self.t2_max = self.t_from_phi(self.phi_max)

		self.coef, self.k = self.fit_eSPS(k=-1.5862)

	def tmax(self, L):
		return (L/2/sin(self.lb/2) - self.l)/self.vd

	def tmin(self, L):
		return (L/2 - self.l)/self.vd

	def _t_from_phi(self, phi):

		l = self.l/self.vd

		A = l/(self.w**2 - 1)
		c = cos(phi/2)

		E = ellipe(asin(self.w*c), 1/self.w)
		F = ellipf(asin(self.w*c), 1/self.w)

		t = 2*l*F/self.w - 3*self.w*E*A - 3*self.w*c*A 

		return float(t)

	def t_from_phi(self, phi):

		return self._t_from_phi(phi) - self.ct

	def phi_from_t(self, t):

		# assert t < self.t2_max

		def t_mismatch(phi):
			dt = t - self.t_from_phi(phi)
			return dt**2

		res = minimize_scalar(t_mismatch, bounds=[self.lb, 2*pi - self.lb], method='bounded')

		return res.x

	def sdsi_ebarrier_point(self, phi, tau):
		""" trajectory under opitmal play, for subgame between invader and the closer defender. 
			reference frame: physical 6D frame, y is along ID_2 upon capture
			----------------------------------------------------------------------------
			Input:
				phi: phi_1 in [lb, ub]									  float
				tau: time spent on the straight line trajectory			 float
				lb:  lower bound of phi.									float
					 lb=LB: at barrier,
					 lb>LB: other SS
			Output:
				x: 4D trajectory, order: D1, I							  np.array
			----------------------------------------------------------------------------
		"""

		l = self.l/self.vd
		w = self.w
		# LB = self.lb
		
		def consts(phi):
			s, c = sin(phi/2), cos(phi/2)
			A, B = l/(w**2 - 1), sqrt(1 - w**2*c**2)
			return s, c, A, B

		# phase II trajectory
		def xtau(phi):

			out = np.zeros((4,))

			s, c, A, B = consts(phi)

			E = float(ellipe(asin(w*c), 1/w))
			F = float(ellipf(asin(w*c), 1/w))

			out[0] = - 2*A*(s**2*B + w*s**3)										# x_D1
			out[1] = A*(w*E + 3*w*c + 2*s*c*B - 2*w*c**3)						   # y_D1
			out[2] = A*(B*(2*w**2*c**2 - w**2 - 1) - 2*w**3*s**3 + 2*w*(w**2-1)*s)  # x_I
			out[3] = A*(w*E + 2*w**2*s*c*B - 2*w**3*c**3 + w*c*(2+w**2))			# y_I
			t = 2*l*F/w - 3*w*E*A - 3*w*c*A  

			return out, t

		s, c, A, B = consts(phi)

		# phase I trajectory plus initial location due to capture range
		xtemp = np.zeros((4,))
		xtemp[0] = -l*sin(self.lb) - 2*s*c*tau							  # x_D1
		xtemp[1] =  l*cos(self.lb) + (2*c**2 - 1)*tau					   # y_D1
		xtemp[2] =			- (w**2*s*c + w*c*B)*tau					  # x_I
		xtemp[3] =			+ (w**2*c**2 - w*s*B)*tau					 # y_I

		xf, tf = xtau(phi)
		x0, t0 = xtau(self.lb)

		dt = max(tf - t0, 0)
		dx = xf - x0
		
		x = dx + xtemp
		t = dt + tau
		
		x = x*self.vd

		# x1 = x[:2]
		# xi = x[2:]
		# print(norm(x1 - xi))

		return x, t

	def tdsi_ebarrier_point(self, phi, tau, gmm, frame='sol'):

		''' input: frame: reference frame 
						  xyL: 
						  sol: the frame for solving the trajectory  
		'''


		assert 0 <= gmm <= pi
		assert (frame == 'xyL' or frame == 'sol')

		x, t = self.sdsi_ebarrier_point(phi, tau)
		d = t*self.vd + self.l

		x_D2 = d*sin(2*gmm - self.lb)
		y_D2 = d*cos(2*gmm - self.lb)

		x = np.array([x[0], x[1], x_D2, y_D2, x[2], x[3]])

		if frame == 'xyL': x = self.to_xyL(x)
		
		return x, t

	def to_xyL(self, x):

		x1 = x[:2]
		x2 = x[2:4]
		xi = x[4:]

		vd = x2 - x1		# vector from D1 to D2
		L = norm(vd)
		ed = vd/L
		ed = np.array([ed[0], ed[1], 0])
		ex = np.array([1, 0, 0])

		sin_a = cross(ex, ed)[-1]
		cos_a = dot(ex, ed)

		# print(sin_a, cos_a)
		C = np.array([[cos_a, sin_a],
					 [-sin_a, cos_a]])

		xm_ = 0.5*(x1 + x2)  # mid point of |D1 D2|

		# translate locations
		xi_ = C.dot(xi - xm_)
		x = np.array([xi_[0], xi_[1], L])

		return x


	def tdsi_ebarrier(self, phi, tau, gmm, n=20, frame='sol'):
		
		assert 0 <= gmm <= pi
		assert tau >= 0

		phi_max = self.get_max_phi(gmm=gmm)
		assert self.lb <= phi <= phi_max
		
		xs, ts = [], []		
		for p in np.linspace(self.lb, phi, n):
			x, t = self.tdsi_ebarrier_point(p, 0, gmm, frame=frame)
			xs.append(x)
			ts.append(t)

		if tau > 0:
			for tau_ in np.linspace(0, tau, 10):
				x_, t = self.tdsi_ebarrier_point(p, tau_, gmm, frame=frame)
				xs.append(x_)
				ts.append(t)

		return np.asarray(xs), np.asarray(t)

	def tdsi_nbarrier_point(self, t, gmm, dlt, frame='phy'):

		assert 2*gmm >= self.lb
		assert 0 <= dlt <= gmm - self.lb/2

		Ld = self.vd*t + self.l
		Li = self.vi*t
		x1 = -Ld*sin(gmm)
		y1 =  Ld*cos(gmm)
		x2 =  Ld*sin(gmm)
		y2 =  Ld*cos(gmm)
		xi = -Li*sin(dlt)
		yi =  Li*cos(dlt)

		x = np.array([x1, y1, x2, y2, xi, yi])

		if frame == 'xyL': x = self.to_xyL(x)

		return x

	def tdsi_nbarrier(self, t, gmm, dlt, n=10, frame='phy'):

		assert 2*gmm >= self.lb
		assert 0 <= dlt <= gmm - self.lb/2

		xs = []
		for t_ in np.linspace(0, t, n):
			x = self.tdsi_nbarrier_point(t_, gmm, dlt, frame=frame)
			xs.append(x)

		return np.asarray(xs), t

	def get_max_phi(self, gmm=None):
		""" obtain the maximum phi
			----------------------------------------------------------------------------
			Input:
				lb:  lower bound of phi.											float
					 lb=LB: at barrier,
					 lb>LB: other SS
			Output:
				ub: (=res.x)														float
					maximum phi is obtained when I, D2, D1 forms a straight line
			----------------------------------------------------------------------------
		"""
		if gmm is None:
			gmm = self.lb/2

		def slope_mismatch(phi):
			x, _ = self.tdsi_ebarrier_point(phi, 0, gmm)
			s_I_D1 = (x[1] - x[5])/(x[0] - x[4])
			s_I_D2 = (x[3] - x[5])/(x[2] - x[4])
			return (s_I_D1 - s_I_D2)**2

		res = minimize_scalar(slope_mismatch, bounds=[pi, 2*pi-self.lb], method='bounded')

		return res.x

	def xd2(self, L, t, xd1):

		# find xd2 = [x, y], the intersection of:
		# C1: x^2 + y^2 = (self.l + self.vd*t)^2
		# C2: (x - xd1)^2 + (y - yd1)^2 = L^2

		d = self.l + self.vd*t

		a = xd1[0]
		b = xd1[1]
		A = a**2 + b**2
		B = A + d**2 - L**2

		D = 4*A*d**2 - B**2

		# there's no solution if 
		# C1 and C2 don't intersect
		if D < 0:
			return None

		x = (a*B - b*sqrt(D))/(2*A)
		y = (B - 2*a*x)/(2*b)

		# print('---------------')
		# print(sqrt(x**2 + y**2), d)
		# print(sqrt((x - a)**2 + (y - b)**2), L)

		return np.array([x, y])


	def solve_eSPS(self, L, gmm, n=10):

		def L_mismatch(tau, L, gmm, phi):
			x, _ = self.tdsi_ebarrier_point(phi, tau, gmm, frame='sol')
			x1 = x[:2]
			x2 = x[2:4]
			xi = x[4:]
			# print(x2)
			# print(xi)
			Ltemp = norm(x1-x2)
			return (Ltemp - L)**2

		tau_max = (L/2/sin(gmm)-self.l)/self.vd

		phi_max = self.get_max_phi(gmm=gmm)

		phis, taus, xyLs = [], [], []
		for p in np.linspace(self.lb, phi_max, n):
			res = minimize_scalar(L_mismatch, 
								  args=(L, gmm, p),
								  bounds=[0, 1.1*tau_max], method='bounded')
			xyL, t = self.tdsi_ebarrier_point(p, res.x, gmm, frame='xyL')
			# print(abs(xyL[-1] - L)/L)
			if abs(xyL[-1] - L)/L > 1e-6:
				break

			phis.append(p)
			taus.append(res.x)
			xyLs.append(xyL)

		return np.asarray(xyLs), np.asarray(phis), np.asarray(taus)
		# phis = np.asarray(phis)
		# taus = np.asarray(taus)
		# for p, t in zip(phis, taus):
		#	 x, t = self.tdsi_ebarrier_point(p, t, gmm, frame='xyL')

		#	 print(x, t)
	
	def solve_nSPS(self, L, gmm, n=10):
		delta = gmm - self.lb/2
		t = (L/2/sin(gmm)-self.l)/self.vd

		xyLs = []
		for dd in np.linspace(0, delta, n):	
			xyL = self.tdsi_nbarrier_point(t, gmm, dd, frame='xyL')
			xyLs.append(xyL)
			# print(xyL)

		return np.asarray(xyLs), t

	def solve_SPS(self, L, gmm, n1=10, n2=10):
		xyLs_n, t = g.solve_nSPS(L, gmm, n=n1)
		xyLs_e, phis, taus = g.solve_eSPS(L, gmm, n=n2)

		ts = [t for _ in range(xyLs_n.shape[0])]
		for phi, tau in zip(phis, taus):
			ts.append(self.t_from_phi(phi) + tau)

		ts = np.asarray(ts)
		xyL = np.concatenate((xyLs_n, xyLs_e), 0)
		return xyL, ts

	# def fit_SPS_Lgmm(self, L, gmm, n1=20, n2=20, order=4):
		
	#	 order = order + 1

	#	 xyL, ts = g.solve_SPS(L, gmm, n1=n1, n2=n2)
	#	 x = xyL[:,0]
	#	 y = xyL[:,1]

	#	 ang, rho = [], []
	#	 for xx, yy in zip(x, y):
	#		 ang.append(atan2(yy,xx))
	#		 rho.append(sqrt(xx**2 + yy**2))
	#	 ang = np.asarray(ang)
	#	 rho = np.asarray(rho)

	#	 apower = [np.array([a**o for a in ang]) for o in range(1, order)]

	#	 data = np.vstack(apower).transpose()

	#	 reg = linear_model.LinearRegression()
	#	 reg.fit(data, rho)
	#	 w0 = reg.intercept_
	#	 ws = reg.coef_
	#	 wr = [w0]+list(ws)

	#	 def get_rho(ang, order=order):
	#		 apower = [ang**o for o in range(1, order)]
	#		 return reg.predict([apower]).squeeze()

	#	 return get_rho, wr

	def get_rho0_tht0(self, L, gmm):

		p = L/2/tan(gmm)
		rho0 = (L/2/sin(gmm) - self.l)*self.w
		tht0 = -(gmm-self.lb/2)

		return p, tht0, rho0

	def en_boundary_point(self, L, gmm):

		p, tht, rho = self.get_rho0_tht0(L, gmm)
		x = rho*sin(tht)
		y = rho*cos(tht) - p

		return x, y

	def en_boundary(self, L, n=20):

		gmms = np.linspace(self.lb/2, pi/2, n)

		xs, ys = [], []
		for gmm in gmms:
			x, y = self.en_boundary_point(L, gmm)
			xs.append(x)
			ys.append(y)

		return np.asarray(xs), np.asarray(ys)

	def fit_eSPS_Lgmm(self, L, gmm, n=20):

		''' given L and gmm, fit the SPS (in polar coordinate)
		'''

		p = L/2/tan(gmm)

		xyL, _, _ = g.solve_eSPS(L, gmm, n=n)
		x = xyL[:,0]
		y = xyL[:,1]

		ang, rho = [], []
		for xx, yy in zip(x, y):
			dx, dy = xx, yy + p
			ang.append((atan2(dx,dy)))
			rho.append((sqrt(dx**2 + dy**2)))
		ang = np.asarray(ang)
		rho = np.asarray(rho)

		# center to ang[0], rho[0]
		ang0, rho0 = ang[0], rho[0]
		ang_ = np.asarray([aa - ang0 for aa in ang])
		rho_ = np.asarray([rr - rho0 for rr in rho])		

		reg = linear_model.LinearRegression(fit_intercept=False)
		reg.fit(ang_.reshape(-1,1), rho_)
		w = reg.coef_

		return w

	def fit_eSPS_L(self, L, ng=20, nxy=100):

		gmms = np.linspace(g.lb/2+.18, pi/2, ng)

		ws = []
		for gmm in gmms:
			w = g.fit_eSPS_Lgmm(L, gmm, n=nxy)
			ws.append(w)
		ws = np.asarray(ws)

		gpower = [np.array([gm**o for gm in gmms]) for o in [1, 2]]
		data = np.vstack(gpower).transpose()

		reg = linear_model.LinearRegression()
		reg.fit(data, ws)
		w0 = float(reg.intercept_)
		w1, w2 = reg.coef_[0]

		# w_fit = np.array([w0 + w1*gm + w2*gm**2 for gm in gmms])
		# plt.plot(gmms, ws, 'o')
		# plt.plot(gmms, w_fit, '--')

		# plt.savefig('wfit_quality_%.2f.png'%L)
		# plt.close()

		return w0, w1, w2

	def fit_eSPS(self, k=-1.5862):
		
		import os

		# =============== if the data exists, read
		if os.path.isfile('bases/hgdata/coefs.npy') and os.path.isfile('bases/hgdata/k.npy'):
			print('reading SPS fitting data....')
			coefs = np.load('bases/hgdata/coefs.npy')
			k = np.load('bases/hgdata/k.npy')

		# =============== otherwise compute
		else:
			# ---------- if the intermedate data for L_wgs exists, read
			if os.path.isfile('bases/hgdata/L_wgs.npy'):
				print('reading L_wgs data....')
				L_wgs = np.load('bases/hgdata/L_wgs.npy')

			# ---------- otherwise compute
			else:
				print('computing L_wgs data....')
				Ls = np.linspace(4, 10, 10)
				wgs = [[], [], []]

				# for each given L, w = wgs[0] + wgs[0]*gm + wgs[0]*gm**2
				for L in Ls:
					wgs_ = self.fit_eSPS_L(L, ng=5)
					for wg, wg_ in zip(wgs, wgs_):
						wg.append(wg_)
					# print(wgs)
				
				for i in range(3):
					wgs[i] = np.asarray(wgs[i])

				L_wgs = np.asarray(np.vstack([Ls] + wgs))
				np.save('bases/hgdata/L_wgs.npy', L_wgs)

			Ls = L_wgs[0]
			wgs = L_wgs[1:]
			
			coefs = []
			for i in range(3):

				label = wgs[i]
				data = np.vstack((np.array([Ls]),
								np.array([exp(k*LL) for LL in Ls]))).transpose()

				reg = linear_model.LinearRegression()
				reg.fit(data, label)
				# scoree += reg.score(data, label)

				w0 = reg.intercept_.squeeze()
				w1, w2 = reg.coef_.squeeze()
				coefs.append([float(w0), float(w1), float(w2)])
				# print(coefs)
		
				# ws_fit = np.asarray([w0 + w1*LL + w2*exp(k*LL) for LL in Ls])
				# plt.plot(Ls, ws, c=colors[i])
				# plt.plot(Ls, ws_fit, '--', c=colors[i])

			coefs = np.asarray(coefs)
			np.save('bases/hgdata/coefs.npy', coefs)
			np.save('bases/hgdata/k.npy', k)

		return coefs, k

	def SPS_fn(self, theta, L, gmm, out='xy'):	 

		assert out == 'xy' or out == 'rho'   
		Lvec = np.array([1, L, exp(self.k*L)])
		gvec = np.array([1, gmm, gmm**2])

		# wg = []
		# for w_ in self.coef:
		#	 wg.append(np.dot(w_, Lvec))
		# wg = np.asarray(wg)
		wg = dot_xA(Lvec, self.coef)

		wsps = dot(wg, gvec)

		p, tht0, rho0 = self.get_rho0_tht0(L, gmm)

		rho = (theta - tht0)*wsps + rho0

		if out == 'rho':
			return rho

		if out == 'xy':
			x = rho*sin(theta)
			y = rho*cos(theta) - p
			return x, y

	def which_SPS(self, L, x, y):

		def en_boundary_mismatch(gmm):
			p, tht, rho = self.get_rho0_tht0(L, gmm)

			# x_ = rho*sin(tht)
			y_ = rho*cos(tht) - p

			return (y_ - y)**2

		res = minimize_scalar(en_boundary_mismatch, bounds=[self.lb/2, pi/2], method='bounded')

		p, tht, rho = self.get_rho0_tht0(L, res.x)

		x_ = rho*sin(tht)
		y_ = rho*cos(tht) - p

		if x < x_:
			return 'envelope'
		else:
			return 'natual'

	def solve_gmm(self, L, x, y):

		assert self.which_SPS(L, x, y) == 'envelope'

		def rho_mismatch(gmm):
			p, tht0, rho0 = self.get_rho0_tht0(L, gmm)
			tht = atan2(x, y+p)
			rho = sqrt(x**2 + (y+p)**2)

			err = rho - self.SPS_fn(tht, L, gmm, out='rho')
			return err**2

		res = minimize_scalar(rho_mismatch, bounds=[self.lb/2, pi/2], method='bounded')

		return res.x
	
	def SPS_df(self, x, y, L, gmm):

		Lvec = np.array([1, L, exp(self.k*L)])
		gvec = np.array([1, gmm, gmm**2])
		dLvec = np.array([0, 1, self.k*exp(self.k*L)])
		dgvec = np.array([0, 1, 2*gmm])

		p, tht0, rho0 = self.get_rho0_tht0(L, gmm)
		tht = atan2(x, y+p)
		rho = sqrt(x**2 + (y+p)**2)

		drho0_dL = self.w/2/sin(gmm)
		drho0_dgm = -L*cos(gmm)/2/(sin(gmm))**2
		dtht0_dgm = -1

		drho_dgm = -L*cos(tht)/2/(sin(gmm))**2
		dtht_dgm =  L*sin(tht)/2/rho/(sin(gmm))**2
		drho_dx = sin(tht)
		dtht_dx = cos(tht)/rho
		drho_dy = cos(tht)
		dtht_dy = -sin(tht)/rho

		fL = dot(dot_xA(dLvec, self.coef), gvec)*(tht - tht0) + drho0_dL
		fg = dot(dot_xA(Lvec, self.coef), dgvec)*(tht - tht0) + \
			 dot(dot_xA(Lvec, self.coef), gvec)*(dtht_dgm - dtht0_dgm) + \
			 drho0_dgm - drho_dgm
		fx = dot(dot_xA(Lvec, self.coef), gvec)*dtht_dx - drho_dx
		fy = dot(dot_xA(Lvec, self.coef), gvec)*dtht_dy - drho_dy

		# p1 = np.array([fx/2 + fL, fy/2 + (y*fx - x*fy)/L])
		# p2 = np.array([fx/2 - fL, fy/2 - (y*fx - x*fy)/L])
		# p1 = p1/norm(p1)
		# p2 = p2/norm(p2)

		return fL, fg, fx, fy

# (-2.4, 1.25)

if __name__ == '__main__':
	
	colors = ['m', 'r', 'k', 'b', 'c']*5
	g = TDSISDHagdorn(1, 1, 1.2)

	L = 7

	xx, yy = -2.4, 1.25

	# verify SPS fitting
	gmms = np.linspace(g.lb/2+.18, pi/2, 4)
	GMM = g.solve_gmm(L, xx, yy)
	gmms = np.concatenate((gmms, np.array([GMM])))
	print(g.SPS_df(xx, yy, L, GMM))
	
	for gmm, c in zip(gmms, colors):

		# the fitted curve
		x_fit, y_fit = [], []
		for ang_ in np.linspace(-(gmm-g.lb/2), -pi/2, 30):
			x, y = g.SPS_fn(ang_, L, gmm)
			x_fit.append(x)
			y_fit.append(y)

		# the actual curve
		xyL, _ = g.solve_SPS(L, gmm, n1=30, n2=30)

		# check SPS fitting
		plt.plot(x_fit, y_fit, '--', c=c)
		plt.plot(xyL[:,0], xyL[:,1], '-', c=c)

	plt.plot(xx, yy, 'o')

	xbd, ybd = g.en_boundary(L)
	plt.plot(xbd, ybd, '-', c='r')

	plt.show()