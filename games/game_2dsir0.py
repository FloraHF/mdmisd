import os
import numpy as np
from math import atan2, acos

from util import norm
from base_2dsir0 import IsochronPointCap
from base_2dsir0 import strategy_pass, strategy_barrier

import rendering as rd

dc = [np.array([0.1, 1., 0.05]),
		np.array([0.4, 0.05, 1.]),
		np.array([0.8, 0.05, 1.]),
		np.array([0.05, 0.5, 0.1]),
		np.array([.7, 1., .5])]
ic = np.array([0.1, 1., 1.])

class TDSISDPointCapPlayer(object):

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

class TDSISDPointCapDPair(object):
	"""docstring for TDSISDPointCapDPair"""
	def __init__(self, D1, D2, vi):

		self.name = 'dpair:%s+%s'%(D1.name, D2.name)
		self.D1 = D1
		self.D2 = D2
		self.Is = dict()

		self.vd = D1.vmax
		self.vi = vi

		self.isc = IsochronPointCap(D1.x, D2.x, self.vd, vi) 

		self.L0 = self.isc.L

		self.color = (D1.color + D2.color)/2

	def get_isc_tranform(self):
		# isc = IsochronPointCap(self.D1.x, self.D2.x, self.vd, self.vi)

		# itemp = [d for k, d in self.Is.items()][0]
		# scale = itemp['t_']/itemp['t']

		scale = self.isc.L/self.L0
		dx = self.isc.xm + \
			np.linalg.inv(self.isc.C)@np.array([-self.isc.L/2, 0]) # move to the middle
		theta = self.isc.theta

		return scale, dx, theta

	def get_barrier_data(self):
		xs, ys = self.isc.barrier_data()
		vs = [(x+self.isc.L/2, y) for x, y in zip(xs, ys)]
		return vs

	def get_isochron_data(self, **arg):

		if 'tpct' in arg and 0<= arg['tpct'] <= 1:
			# assert 0<= arg['t'] <= 1
			tmin, tmax = self.isc.trange()
			t = arg['tpct']*tmin + (1-arg['tpct'])*tmax
		
		elif 't' in arg:
			tmin, tmax = self.isc.trange()
			if tmin <= arg['t'] <= tmax:
				t = arg['t']
			else:
				if arg['t'] < tmin:
					t = -1
				if arg['t'] > tmax:
					t = np.infty

		elif 'xy' in arg:
			t = self.isc.get_t(arg['xy'][0], arg['xy'][1])

		if 0 < t < np.infty:
			xs, ys = self.isc.isochron_data(t)
			vs = [(x+self.isc.L/2, y) for x, y in zip(xs, ys)]
			return vs		

		return None

	def update(self):
		self.isc.update(self.D1.x, self.D2.x)
		for i, p in self.Is.items():
			p['t'] = p['t_']
			x, y = self.isc.get_xy(p['p'].x)
			p['t_'] = self.isc.get_t(x, y)

	def add_invader(self, I):
		x, y = self.isc.get_xy(I.x)
		t = self.isc.get_t(x, y)
		self.Is.update({I.name:{'p':I, 't':t, 't_':t}})

	def reset(self):
		self.isc.update(self.D1.x0, self.D2.x0)

class TDSISDPointCapGame():
	"""docstring for TDSISDPointCapGame"""
	def __init__(self, vd, vi, dt=.01,
						AR=1):		

		self.vd = vd
		self.vi = vi
		self.a = vi/vd
		self.gmm = acos(vd/vi)

		self.t = 0.
		self.dt = dt

		# players
		L, x = 10, 4
		self.D1 = TDSISDPointCapPlayer(np.array([-L/2, 0.]), 
										vd, dt=dt, name='D1', color=dc[0])
		self.D2 = TDSISDPointCapPlayer(np.array([L/2, 0.]), 
										vd, dt=dt, name='D2', color=dc[1])

		isc = IsochronPointCap(self.D1.x, self.D2.x, vd, vi)
		ymin, ymax = isc.yrange(x)
		self.I  = TDSISDPointCapPlayer(np.array([x, .5*(ymin+ymax)]), 
										vd, dt=dt, name='I', color=ic)
		self.players = [self.D1, self.D2, self.I]

		# isochrons for defender pairs
		dpairs_temp = [(self.D1, self.D2)]
		self.dpairs = [TDSISDPointCapDPair(self.D1, self.D2, vi) 
						for dp in dpairs_temp]
		self.dpairs[0].add_invader(self.I)
		for p in self.dpairs:
				p.update()

		# for rendering
		self.AR = AR
		self._reset_render()						 

	def iscap(self, x1, x2, xi):

		if norm(x1 - xi)<.1 or norm(x2 - xi)<.1:
			return True

		return False

	def isfeasible(self, x1, x2, xi):
		feasible = False
		for p in self.dpairs:
			x, y = p.isc.get_xy(xi)
			t = p.isc.get_t(x, y)
			feasible = feasible or 0 < t < np.infty
		return feasible

	def play(self, dstr=strategy_barrier, 
					istr=strategy_barrier, 
					render=False,
					record=False):

		if render:
			self.viewer = rd.Viewer(self.AR*700, 700)

		if record:
			if os.path.exists('time.csv'):
				os.remove('time.csv')

		xs = [p.x for p in self.players]
		x1s, x2s, xis = [xs[0]], [xs[1]], [xs[2]]

		for i in range(1500):

			# print('--------------------', i, self.t, '--------------------')

			str_in = xs + [self.vd, self.vi]
			v1, v2, _ = dstr(*str_in, record=record)
			_, _, vi  = istr(*str_in)
			vs = [v1, v2, vi]

			# take one step
			for p, v in zip(self.players, vs):
				p.step(v)
			self.t += self.dt

			for p in self.dpairs:
				p.update()
				# print(p.isc.theta)

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

			if not self.isfeasible(*xs):
				break

		print('%s(D) v.s %s(I), x_c=[%.2f, %.2f], t=%.2f'%
			  (dstr, istr, self.I.x[0], self.I.x[1], self.t))

		return np.asarray(x1s), np.asarray(x2s), np.asarray(xis)

	def reset(self):
		for p in self.players:
			p.reset()
		self.t = 0.

	# reset rendering assets
	def _reset_render(self):
		self.render_geoms = None
		self.render_geoms_xform = None
		self.render_isochrons = None
		self.render_isochrons_xform = None

	def render(self):
		# create geoms if doesn't exist
		if self.render_geoms is None:

			self.render_geoms = {}
			self.render_geoms_xform = {}

			for p in self.players:

				# body
				res = 3 if 'I' in p.name else 30
				body = rd.make_circle(p.size, res=res, filled=True)
				body.set_color(*p.color)

				geom = {p.name+':body': body}

				# transform
				xform = rd.Transform()
				
				for key, g in geom.items(): # all that in geom share the same transform
					g.add_attr(xform)

				self.render_geoms.update(geom)
				self.render_geoms_xform.update({p.name: xform})			

			for p in self.dpairs:

				geom = {}
				xform = rd.Transform()

				v = p.get_barrier_data()
				barrier = rd.make_polyline(v)
				barrier.set_color(*p.color)
				geom.update({p.name+':barrier':barrier})

				for t in [.2, .9, .999999]:
					v = p.get_isochron_data(tpct=t)
					isochron = rd.make_polyline(v)
					isochron.set_color(*p.color)
					geom.update({p.name+':iscf_%.2f'%t:isochron})

				for key, g in geom.items(): # all that in geom share the same transform
					g.add_attr(xform)

				self.render_geoms.update(geom)
				self.render_geoms_xform.update({p.name: xform})	

			# add geoms to viewer
			self.viewer.geoms = []
			for key, geom in self.render_geoms.items():
				self.viewer.add_geom(geom)

		isc_geoms = {}
		isc_geoms_xforms = {}

		for p in self.dpairs:

			geom = {}
			xform = rd.Transform()

			itemp = [d for k, d in p.Is.items()][0]
			
			# the isochron to chase after
			tau = itemp['t'] - self.dt
			v = p.get_isochron_data(t=itemp['t'] - self.dt)
			if v is not None:
				isochron = rd.make_polyline(v)
				isochron.set_color(*p.color)
				geom.update({p.name+':isc_%.2f'%tau:isochron})

			for key, g in geom.items(): # all that in geom share the same transform
				g.add_attr(xform)

			isc_geoms.update(geom)
			isc_geoms_xforms.update({p.name: xform})	

		for key, geom in isc_geoms.items():
			self.viewer.add_onetime(geom)
		
		# render: set viewer window bounds
		xm, ym, h = -1., 1, 15
		xl, xr = xm - self.AR*h/2,  xm + self.AR*h/2
		yb, yt = ym - h/2,			ym + h/2
		self.viewer.set_bounds(xl, xr, yb, yt)		

		# render: players
		for p in self.players:
			self.render_geoms_xform[p.name].set_translation(*p.x)

		# render: isochrons
		for p in self.dpairs:

			scale, translation, rotation = p.get_isc_tranform()

			self.render_geoms_xform[p.name].set_translation(*translation)
			self.render_geoms_xform[p.name].set_rotation(rotation)
			self.render_geoms_xform[p.name].set_scale(scale, scale)

			isc_geoms_xforms[p.name].set_translation(*translation)
			isc_geoms_xforms[p.name].set_rotation(rotation)

		self.viewer.render()