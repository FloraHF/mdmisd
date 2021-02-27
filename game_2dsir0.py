import numpy as np
from math import atan2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.collections as mcol

from matplotlib import rc
from matplotlib.legend_handler import HandlerTuple
rc("text", usetex=True)

from util import norm
from base_2dsir0 import TDSISDPointCap

from handler import HandlerDashedLines

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
	def __init__(self, vd, vi, dt=.02):		
		super(TDSISDPointCapGame, self).__init__(vd, vi)

		self.t = 0.
		self.dt = dt
		L, x = 10, 4
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

		for i in range(275):

			# if dstr == 't':
			# 	v1, v2 = self.strategy[dstr](*xs, self.I.v)
			# else:
			# 	v1, v2, _ = self.strategy[dstr](*xs)
			
			# print(dstr, istr)
			v1, v2, _ = self.strategy[dstr](*xs)				
			_, _, vi  = self.strategy[istr](*xs)
			# print('vi out', vi)
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

	fs = 21
	lw = 2


	# ############### change invaders trategy (Figure 6) ###############
	# g = TDSISDPointCapGame(1, 1.2)
	# x1_b, x2_b, xi_b = g.play(dstr='b', istr='b')

	# plt.figure(figsize=(6.8, 4.8))
	# plt.plot(x1_b[:,0], x1_b[:,1], 
	# 			'-o', markevery=70, lw=lw, color='b', label=r'$D, proposed$')
	# plt.plot(x2_b[:,0], x2_b[:,1], 
	# 			'-o', markevery=70, lw=lw, color='b')
	# plt.plot(xi_b[:,0], xi_b[:,1], 
	# 			'-o', markevery=70, lw=lw, color='r', label=r'$I, proposed$')

	# plt.plot(xi_b[-1,0], xi_b[-1,1], 'd', lw=lw, color='r', zorder=100)
	# plt.plot(x1_b[-1,0], x1_b[-1,1], 'o', lw=lw, color='b')
	# plt.plot(x2_b[-1,0], x2_b[-1,1], 'o', lw=lw, color='b')

	# plt.text(xi_b[0,0], xi_b[0,1]+.2, r'$I$', fontsize=fs*.9)
	# plt.text(x1_b[0,0], x1_b[0,1]-.6, r'$D_1$', fontsize=fs*.9)
	# plt.text(x2_b[0,0]-.1, x2_b[0,1]-.6, r'$D_2$', fontsize=fs*.9)

	# g.reset()
	# x1_p, x2_p, xi_p = g.play(dstr='b', istr='p')

	# plt.plot(x1_p[:,0], x1_p[:,1], 
	# 			'--o', markevery=70, lw=lw, color='b', alpha=.75, label=r'$D, proposed$')
	# plt.plot(x2_p[:,0], x2_p[:,1], 
	# 			'--o', markevery=70, lw=lw, color='b', alpha=.75)
	# plt.plot(xi_p[:,0], xi_p[:,1], 
	# 			'--o', markevery=70, lw=lw, color='r', alpha=.75, label=r'$I, Strategy (26b)$')
	# plt.plot(xi_p[-1,0], xi_p[-1,1], 'd', lw=lw, color='r', zorder=100)
	# plt.plot(x1_p[-1,0], x1_p[-1,1], 'o', lw=lw, color='b')
	# plt.plot(x2_p[-1,0], x2_p[-1,1], 'o', lw=lw, color='b')

	# plt.gca().add_patch(plt.Circle((0, -25), 23, ec='b', fc='lightsteelblue', lw=2))
	# plt.text(-.6, -3, r'$Target$', fontsize=fs*.9)

	# # plt.legend(fontsize=fs*0.8)

	# line = [[(0, 0)]]
	# lc_opt = mcol.LineCollection(2*line, 
	# 							linestyles=['-', '-'], 
	# 							colors=['b', 'r'], 
	# 							linewidths=[2, 2])
	# lc_sub = mcol.LineCollection(2*line, 
	# 							linestyles=[(0,(3,1)), (0,(3,1))], 
	# 							colors=['b', 'r'], 
	# 							linewidths=[2, 2])

	# plt.gca().legend([lc_opt, lc_sub], ['(16) vs. (17)', '(16) vs. (26b)'], 
	# 			handler_map={type(lc_opt): HandlerDashedLines()},
	#           	handlelength=2.5, handleheight=3, 
	#           	fontsize=fs*.9)


	# plt.xlabel(r'$x (m)$', fontsize=fs)
	# plt.ylabel(r'$y (m)$', fontsize=fs)

	# plt.gca().tick_params(axis="both", which="major", labelsize=fs)
	# plt.gca().tick_params(axis="both", which="minor", labelsize=fs)
	# plt.subplots_adjust(bottom=.17, top=0.95, left=.13, right=0.96)
	# plt.grid()
	# plt.axis('equal')
	# plt.xlim((-6, 6))
	# plt.ylim((-3, 4))
	# plt.show()

	############### change invaders trategy (Figure 7) ###############
	g = TDSISDPointCapGame(1, 1.2)
	x1_b, x2_b, xi_b = g.play(dstr='b', istr='b')

	plt.figure(figsize=(6.8, 5.8))
	plt.plot(x1_b[:,0], x1_b[:,1], 
				'-o', markevery=70, lw=lw, color='b', label=r'$D, proposed$')
	plt.plot(x2_b[:,0], x2_b[:,1], 
				'-o', markevery=70, lw=lw, color='b')
	plt.plot(xi_b[:,0], xi_b[:,1], 
				'-o', markevery=70, lw=lw, color='r', label=r'$I, proposed$')

	plt.plot(xi_b[-1,0], xi_b[-1,1], 'd', lw=lw, color='r', zorder=100)
	plt.plot(x1_b[-1,0], x1_b[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_b[-1,0], x2_b[-1,1], 'o', lw=lw, color='b')

	plt.text(xi_b[0,0], xi_b[0,1]+.2, r'$I$', fontsize=fs*.9)
	plt.text(x1_b[0,0]-.55, x1_b[0,1]-.6, r'$D_1$', fontsize=fs*.9)
	plt.text(x2_b[0,0]-.1, x2_b[0,1]-.6, r'$D_2$', fontsize=fs*.9)

	g.reset()
	x1_p, x2_p, xi_p = g.play(dstr='p', istr='b')

	plt.plot(x1_p[:,0], x1_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75, label=r'$D, proposed$')
	plt.plot(x2_p[:,0], x2_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75)
	plt.plot(xi_p[:,0], xi_p[:,1], 
				'--o', markevery=70, lw=lw, color='r', alpha=.75, label=r'$I, Eq.(53)$')
	plt.plot(xi_p[-1,0], xi_p[-1,1], 'd', lw=lw, color='r', zorder=100)
	plt.plot(x1_p[-1,0], x1_p[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_p[-1,0], x2_p[-1,1], 'o', lw=lw, color='b')

	plt.gca().add_patch(plt.Circle((0, -25), 23, ec='b', fc='lightsteelblue', lw=2))
	plt.text(-.6, -4, r'$Target$', fontsize=fs*.9)

	# plt.legend(fontsize=fs*0.8)

	line = [[(0, 0)]]
	lc_opt = mcol.LineCollection(2*line, 
								linestyles=['-', '-'], 
								colors=['b', 'r'], 
								linewidths=[2, 2])
	lc_sub = mcol.LineCollection(2*line, 
								linestyles=[(0,(3,1)), (0,(3,1))], 
								colors=['b', 'r'], 
								linewidths=[2, 2])

	plt.gca().legend([lc_opt, lc_sub], ['(16)  vs. (17)', '(26a) vs. (17)'], 
				handler_map={type(lc_opt): HandlerDashedLines()},
	          	handlelength=2.5, handleheight=3, 
	          	fontsize=fs*.9)


	plt.xlabel(r'$x (m)$', fontsize=fs)
	plt.ylabel(r'$y (m)$', fontsize=fs)

	plt.gca().tick_params(axis="both", which="major", labelsize=fs)
	plt.gca().tick_params(axis="both", which="minor", labelsize=fs)
	plt.subplots_adjust(bottom=.15, top=0.95, left=.13, right=0.96)
	plt.grid()
	plt.axis('equal')
	plt.xlim((-6, 6))
	plt.ylim((-3, 4))
	plt.show()