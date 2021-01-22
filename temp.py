import numpy as np
from math import sqrt, pi, sin, cos, atan2
from scipy.integrate import quad

import rendering as rd
from util import dot, norm, dist

dcolors = [np.array([0.1, 1., 0.05]),
		   np.array([0.4, 0.05, 1.]),
		   np.array([0.8, 0.05, 1.]),
		   np.array([0.05, 0.5, 0.1]),
		   np.array([.7, 1., .5])]
icolor = np.array([0.1, 1., 1.])

class MDSISDLinePlayer(object):
	""" Base player for the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, r, a, nd, dt=0.1, # for game
				 name='', 				 # for rendering
				 size=.1,
				 color=[1, 1, 1],):
		
		# inputs
		self.param = MDSISDLineParam(r, a, nd)
		self.dt = dt

		# for rendering
		self.name = name
		self.size = size
		self.color = color

	def reset(self):
		self.x = np.array([xx for xx in self.x0])
		self.v = np.array([0, 0])
		
	def step(self, v):
		self.v = np.array([vv for vv in v])
		# print(type(self.dt))
		self.x = self.x + self.v*self.dt		

class MDSISDLineDefender(MDSISDLinePlayer):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
		
	def __init__(self, j, *arg, **kwarg):

		super(MDSISDLineDefender, self).__init__(*arg, **kwarg)
		self.id = j

		# segment responsible [D_l, D_r]
		self.D_l = self.param.D[j]
		self.D_r = self.param.D[j+1]

		# initial location
		self.x0 = np.array([self.param.xd[j], 0])

		# critical points on the line
		self.a_r = self.param.a_r[j]
		self.c_r = self.param.c_r[j]
		self.a_l = self.param.a_l[j] 
		self.c_l = self.param.c_l[j]

		# state
		self.reset()

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
			if self.in_CR(xi):
				vx = dot(self.param.e_R, vi)/self.param.a
				# print('defender ', self.id, 'in CR, vx=', vx)
			elif self.in_CL(xi):
				vx = dot(self.param.e_L, vi)/self.param.a
				# print('defender ', self.id, 'in CL, vx=', vx)
			else:
				vx = 1.
				# print('defender ', self.id, 'in Omega, vx=', vx)
		else:
			# print('defender ', self.id, 'rest')
			vx = 0.

		return np.array([vx, 0])

class MDSISDLineIntruder(MDSISDLinePlayer):
	""" Defenders of the multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
	"""
	def __init__(self, *arg, **kwarg):

		super(MDSISDLineIntruder, self).__init__(*arg, **kwarg)

		# state
		self.x0 = np.array([0, 0]) 
		self.reset()

	def get_w(self, z, j):
		# see Fig.1 in the paper
		e = z/norm(z)
		t = atan2(e[1], e[0])
		# print(t, self.param.beta)

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


	def strategy_x(self, xds):
		xd, jd = self.get_locald(xds)
		w, Rw = self.get_w(self.x - xd, jd)

		vd = np.array([1, 0])
		vd_w = dot(vd, w)
		vd_Rw = dot(vd, Rw)

		vi_Rw = vd_Rw
		vi_w = sqrt(self.param.a**2 - vi_Rw**2)

		vi = vi_Rw*Rw + vi_w*w

		# print(w, Rw)
		
		return vi

class MDSISDLineGame(object):
	""" Parameters for multi-defender single-intruder
		guarding a line segment game with slower defenders.
		See 2009, Witold Rzymowski, "A problem of guarding a line segment"
		input a: speed ratio vi/vd
			  r: capture range of the defender
	"""
	def __init__(self, r, a, nd, AR=3):

		# inputs:
		self.param = MDSISDLineParam(r, a, nd)

		# players:
		self.intruder = MDSISDLineIntruder(r, a, nd, 
											name='I0',
											color=icolor)
		self.defenders = [MDSISDLineDefender(j, r, a, nd, 
												name='D'+str(j),
												color=dcolors[j]) for j in range(nd)]

		# for rendering:
		self.AR = AR
		self.viewer = rd.Viewer(self.AR*300, 300)
		self._reset_render()

	# main iteration, play the game
	def play(self):
		xds = [d.x for d in self.defenders]
		vi = self.intruder.strategy_x(xds)
		self.intruder.step(vi)
		for d in self.defenders:
			vd = d.strategy_v(self.intruder.x, self.intruder.v)
			# print(vd)
			d.step(vd)
			self.render()

	# reset rendering assets
	def _reset_render(self):
		self.render_geoms = None
		self.render_geoms_xform = None

	def render(self):
		# create geoms if doesn't exist
		if self.render_geoms is None:
			self.render_geoms = {}
			self.render_geoms_xform = {}
			for p in [self.intruder] + self.defenders:
				res = 3 if 'I' in p.name else 30
				# body
				body = rd.make_circle(p.size, res=res, filled=True)
				# print(p.color)
				color = p.color
				body.set_color(*color)
				geom = {p.name+':body': body}
				xform = rd.Transform()
				
				for key, g in geom.items(): # all that in geom share the same transform
					g.add_attr(xform)

				self.render_geoms.update(geom)
				self.render_geoms_xform.update({p.name: xform})

			# add geoms to viewer
			self.viewer.geoms = []
			for key, geom in self.render_geoms.items():
				self.viewer.add_geom(geom)
		
		# render
		h = 5
		self.viewer.set_bounds(-2, self.AR*h-2, -1, h-1)		
		for p in [self.intruder] + self.defenders:
			# print(p.name, p.x)
			self.render_geoms_xform[p.name].set_translation(*p.x)

		self.viewer.render()


if __name__ == '__main__':
	g = MDSISDLineGame(1, 1.5, 5)
	# print(g.D)
	for i in range(200):
		g.play()
