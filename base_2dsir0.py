import numpy as np
from math import sqrt, sin, cos, acos, atan2, pi
import cvxpy as cp

from util import dot, cross, norm

class IsochronPointCap(object):
	"""docstring for Isochron"""
	def __init__(self, x1, x2, vd, vi):

		self.x1 = x1
		self.x2 = x2

		self.vi = vi
		self.vd = vd
		self.a = vi/vd
		self.gmm = acos(vd/vi)

		self.L, self.xm, self.C, self.theta = self.get_transform()

	def get_transform(self):

		# translation
		xm = 0.5*(self.x1 + self.x2)

		# L
		vd = self.x2 - self.x1
		L = norm(vd)

		# rotation
		ed = vd/L
		ed = np.array([ed[0], ed[1], 0])
		ex = np.array([1, 0, 0])

		sin_a = cross(ex, ed)[-1]
		cos_a = dot(ex, ed)
		C = np.array([[cos_a, sin_a],
					 [-sin_a, cos_a]])

		return L, xm, C, atan2(sin_a, cos_a)

	def get_xy(self, xi):
		xi = self.C.dot(xi - self.xm)
		x = xi[0]
		y = xi[1]

		return x, y

	def trange(self):
		tmin = self.L/2/self.vd
		tmax = self.L/2/sin(self.gmm)/self.vd
		return tmin, tmax

	def yrange(self, x):	
		ymax = sqrt((self.vi*self.L/(2*self.vd))**2 - x**2)
		ymin = sqrt((self.a*self.L/2)**2 - x**2)*sin(self.gmm)
		return ymin, ymax
		
	def get_t(self, x, y):

		# print(x, y, L)

		if sqrt(x**2 + y**2)/self.vi > self.L/2/self.vd: 
			# print('invader far away')
			return -1 # don't have to take action

		D = y**2 - (self.a**2 - 1)*((self.a*self.L/2)**2 - (x**2 + y**2))
		if D < 0:			# the invader is below the barrier
			# print('invader below barrier')
			return np.infty # can't capture
		
		# print('take action!')
		p = (y - sqrt(D))/(self.a**2 - 1)
		t = sqrt(p**2 + (self.L/2)**2)/self.vd
		
		return t	
		
	def p(self, L, t) :
		return sqrt((self.vd*t)**2 - (L/2)**2)

	def f(self, x, y, L, t):
		return x**2 + (y + self.p(L, t))**2 - (self.vi*t)**2

	def fx(self, x, y, L, t):
		return 2*x

	def fy(self, x, y, L, t):
		return 2*(y + self.p(L, t))

	def fL(self, x, y, L, t):
		return -.5*L*(y/self.p(L, t) + 1)		
	
	def ft(self, x, y, L, t):
		return 2*self.vd**2*t*(y/self.p(L, t) + 1) \
				- 2*self.vi**2*t

	def ee(self, x, y, L, t):
		fx = self.fx(x, y, L, t)
		fy = self.fy(x, y, L, t)
		den = sqrt(fx**2 + fy**2)
		return -np.array([fx, fy])/den

	def pp(self, x, y, L, t):
		fx = self.fx(x, y, L, t)
		fy = self.fy(x, y, L, t)
		fL = self.fL(x, y, L, t)
		yfx_xfy = y*fx - x*fy
		p1 = np.array([fx/2 + fL, fy/2 + yfx_xfy/L])
		p2 = np.array([fx/2 - fL, fy/2 - yfx_xfy/L])
		return p1, p2

	def Jbar(self, v1, v2, vi, x, y, L, t):
		fx = self.fx(x, y, L, t)
		fy = self.fy(x, y, L, t)
		fL = self.fL(x, y, L, t)
		ft = self.ft(x, y, L, t)

		p1, p2 = self.pp(x, y, L, t)
		ee = self.ee(x, y, L, t)

		den = sqrt(fx**2 + fy**2)
		J1 = dot(p1, v1)/den
		J2 = dot(p2, v2)/den
		JI = dot(vi, ee)
		Jc = ft/den# constant

		return J1 + J2 + JI + Jc

	def isochron_data(self, t, n=50):
		xmax = self.L/2*(self.vi/self.vd)
		xs = np.linspace(-xmax, xmax, n)
		ys = np.array([sqrt((self.vi*t)**2 - x**2) - 
						sqrt((self.vd*t)**2 - (self.L/2)**2) for x in xs])
		return xs, ys

	def barrier_data(self, n=50):
		xmax = self.L/2*(self.vi/self.vd)

		xs = np.linspace(-xmax, xmax, n)
		ys = np.array([sqrt((xmax**2 - x**2)*(1 - (self.vd/self.vi)**2))
							for x in xs])

		return xs, ys

	def update(self, x1, x2):
		self.x1 = np.array([x for x in x1])
		self.x2 = np.array([x for x in x2])
		self.L, self.xm, self.C, self.theta = self.get_transform()

def strategy_pass(x1, x2, xi, vd, vi):
	isc = IsochronPointCap(x1, x2, vd, vi)
	C = np.linalg.inv(isc.C)

	I_D1 = x1 - xi
	I_D2 = x2 - xi
	D1_I = -I_D1
	D2_I = -I_D2
	tht_1 = atan2(D1_I[1], D1_I[0])
	tht_2 = atan2(D2_I[1], D2_I[0])
	tht_I = atan2(I_D2[1], I_D2[0])

	sin_tht = cross(I_D1, I_D2)[-1]
	cos_tht = dot(I_D1, I_D2)
	den_tht = sqrt(sin_tht**2 + cos_tht**2)
	sin_tht = sin_tht/den_tht
	cos_tht = cos_tht/den_tht
	tht = atan2(sin_tht, cos_tht)

	d1 = norm(I_D1)
	d2 = norm(I_D2)
	k = d1/d2

	phi_1 = -pi/2
	phi_2 = pi/2

	cos_I = sin_tht
	sin_I = cos_tht - k
	phi_I = atan2(sin_I, cos_I)

	p1 = phi_1 + tht_1
	p2 = phi_2 + tht_2
	pI = phi_I + tht_I

	v1 = np.array([cos(p1), sin(p1)])*vd
	v2 = np.array([cos(p2), sin(p2)])*vd
	vi = np.array([cos(pI), sin(pI)])*vi

	return C.dot(v1), C.dot(v2), C.dot(vi)

def strategy_barrier(x1, x2, xi, vd, vi):

	isc = IsochronPointCap(x1, x2, vd, vi)
	C = np.linalg.inv(isc.C)
	L = isc.L
	x, y = isc.get_xy(xi)
	t = isc.get_t(x, y)

	if t == np.infty:
		v1 = np.array([0, 0])
		v2 = np.array([0, 0])
		vi = np.array([0, 0]) # TODO: can't get through

	if t < 0:
		# print('can pass')
		v1, v2, vi = strategy_pass(x1, x2, xi, vd, vi) # TODO: can get through

	if 0 <= t < np.infty:
		e = isc.ee(x, y, L, t)
		p1, p2 = isc.pp(x, y, L, t)

		vi = vi*e
		v1 = -vd*p1/norm(p1)
		v2 = -vd*p2/norm(p2)

	return C.dot(v1), C.dot(v2), C.dot(vi)

# class TDSISDPointCap(object):
# 	"""Two-defender single-intruder game
# 	with fixed Phase 2 defending strategy"""
# 	def __init__(self, vd, vi):
		
# 		self.vi = vi
# 		self.vd = vd
# 		self.a = vi/vd
# 		self.gmm = acos(vd/vi)

# 		# self.isc = IsochronPointCap(vd, vi)

# 	def get_xyL(self, x1, x2, xi):

# 		vd = x2 - x1
# 		L = norm(vd)
# 		ed = vd/L
# 		ed = np.array([ed[0], ed[1], 0])
# 		ex = np.array([1, 0, 0])

# 		sin_a = cross(ex, ed)[-1]
# 		cos_a = dot(ex, ed)
# 		C = np.array([[cos_a, sin_a],
# 					 [-sin_a, cos_a]])

# 		xm = 0.5*(x1 + x2)	# mid point of |D1 D2|

# 		# translate locations
# 		xi = C.dot(xi - xm)
# 		x = xi[0]
# 		y = xi[1]
		
# 		return x, y, L, np.linalg.inv(C)

# 	def strategy_pass(self, x1, x2, xi):
# 		_, _, _, C = self.get_xyL(x1, x2, xi)

# 		I_D1 = x1 - xi
# 		I_D2 = x2 - xi
# 		D1_I = -I_D1
# 		D2_I = -I_D2
# 		tht_1 = atan2(D1_I[1], D1_I[0])
# 		tht_2 = atan2(D2_I[1], D2_I[0])
# 		tht_I = atan2(I_D2[1], I_D2[0])

# 		sin_tht = cross(I_D1, I_D2)[-1]
# 		cos_tht = dot(I_D1, I_D2)
# 		den_tht = sqrt(sin_tht**2 + cos_tht**2)
# 		sin_tht = sin_tht/den_tht
# 		cos_tht = cos_tht/den_tht
# 		tht = atan2(sin_tht, cos_tht)
# 		# print(tht*180/3.14)

# 		d1 = norm(I_D1)
# 		d2 = norm(I_D2)
# 		k = d1/d2

# 		phi_1 = -pi/2
# 		phi_2 = pi/2

# 		cos_I = sin_tht
# 		sin_I = cos_tht - k
# 		phi_I = atan2(sin_I, cos_I)
# 		# print(phi_I*180/3.14)

# 		p1 = phi_1 + tht_1
# 		p2 = phi_2 + tht_2
# 		pI = phi_I + tht_I
# 		# print(pI*180/3.14)

# 		v1 = np.array([cos(p1), sin(p1)])*self.vd
# 		v2 = np.array([cos(p2), sin(p2)])*self.vd
# 		vi = np.array([cos(pI), sin(pI)])*self.vi

# 		vtemp = C.dot(vi)
# 		# print('vi in', C.dot(vi))

# 		return C.dot(v1), C.dot(v2), C.dot(vi)

# 	def strategy_default(self, x1, x2, xi):
# 		x, y, L, C = self.get_xyL(x1, x2, xi)
# 		t = self.isc.get_t(x, y, L)
# 		p = self.isc.p(L, t)
# 		pc = np.array([0, -p])

# 		v1 = pc - x1
# 		v2 = pc - x2
# 		vi = pc - xi
	
# 		v1 = self.vd*v1/norm(v1)
# 		v2 = self.vd*v2/norm(v2)
# 		vi = self.vi*vi/norm(vi)

# 		return C.dot(v1), C.dot(v2), C.dot(vi)

# 	def strategy_barrier(self, x1, x2, xi):

# 		x, y, L, C = self.get_xyL(x1, x2, xi)
# 		t = self.isc.get_t(x, y, L)
# 		# e = self.isc.ee(x, y, L, t)

# 		if t == np.infty:
# 			v1 = np.array([0, 0])
# 			v2 = np.array([0, 0])
# 			vi = np.array([0, 0]) # TODO: can't get through

# 		if t < 0:
# 			# print('can pass')
# 			v1, v2, vi = self.strategy_pass(x1, x2, xi) # TODO: can get through

# 		if 0 <= t < np.infty:
# 			e = self.isc.ee(x, y, L, t)
# 			p1, p2 = self.isc.pp(x, y, L, t)

# 			vi = self.vi*e
# 			v1 = -self.vd*p1/norm(p1)
# 			v2 = -self.vd*p2/norm(p2)


# 			# print(atan2(vi[1], vi[0])*180/pi)

# 		return C.dot(v1), C.dot(v2), C.dot(vi)

# 	def strategy_tdefense(self, x1, x2, xi, vi):
		
# 		x, y, L, C = self.get_xyL(x1, x2, xi)
# 		t = self.isc.get_t(x, y, L)

# 		fx = self.isc.fx(x, y, L, t)
# 		fy = self.isc.fy(x, y, L, t)
# 		fL = self.isc.fL(x, y, L, t)
# 		ft = self.isc.ft(x, y, L, t)
# 		den = sqrt(fx**2 + fy**2)

# 		p1, p2 = self.isc.pp(x, y, L, t)
# 		ee = self.isc.ee(x, y, L, t)

		
# 		F = np.array([0, 1, 0, 1]) 			# maximize v1y + v2y
# 		A1 = np.concatenate((p1, p2))/den  	# for J<= 0
# 		A1 = A1.reshape(1,4)
# 		A2 = np.array([[ 0, 1, 0, 0],
# 					   [ 0, 0, 0, 1]]) 		# for v1y, v2y to be negative
# 		A3 = -fL*np.array([[-1, 0, 1, 0]]) 	# for ee in 4th quadrant

# 		b1 = np.array([-vi.dot(ee) - ft/den])
# 		b2 = np.array([0, 0])
# 		b3 = np.array([-ft])

# 		A = np.concatenate((A1, A2, A3), 0)
# 		b = np.concatenate((b1, b2, b3))

# 		P1 = np.array([[1, 0, 0, 0],
# 					   [0, 1, 0, 0],
# 					   [0, 0, 0, 0],
# 					   [0, 0, 0, 0]])

# 		P2 = np.array([[0, 0, 0, 0],
# 					   [0, 0, 0, 0],
# 					   [0, 0, 1, 0],
# 					   [0, 0, 0, 1]])

# 		# print(cp)
# 		vd = cp.Variable(4)
# 		prob = cp.Problem(cp.Maximize(F@vd), 
# 							[A@vd<=b, 							# J<=0, v1y, v2y<0, ee in 4th quadrant
# 							 cp.quad_form(vd,P1)<=self.vd**2, 	# |v1| <= vd
# 							 cp.quad_form(vd,P2)<=self.vd**2])	# |v2| <= vd

# 		prob.solve()
# 		v1 = vd.value[:2]
# 		v2 = vd.value[2:]

# 		return C.dot(v1), C.dot(v2)


if __name__ == '__main__':
	# g = TDSISDPointCap(1, 1.4)
	x1 = np.array([-5, 0.])
	x2 = np.array([ 5, 0.])
	xi = np.array([.0, 6.])
	vx = .0
	# v1 = np.array([vx,  -sqrt(1-vx**2)])
	# v2 = np.array([-vx, -sqrt(1-vx**2)])
	vi = np.array([-vx, -sqrt(1.4**2-vx**2)])

	# print(g.strategy_barrier(x1, x2, xi))
	# v1d, v2d, vid = g.strategy_default(x1, x2, xi)
	v1p, v2p, vip = strategy_pass(x1, x2, xi, 1, 1.4)

	print(v1p, v2p, vip)
	