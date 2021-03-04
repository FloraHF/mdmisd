import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.collections as mcol

from matplotlib import rc
from matplotlib.legend_handler import HandlerTuple
rc("text", usetex=True)

from game_2dsir0 import TDSISDPointCapGame
from base_2dsir0 import strategy_pass, strategy_barrier
from handler import HandlerDashedLines

fs = 21
lw = 2

################ change invaders trategy (Figure 6) ###############
def compare_dstrategy():
	g = TDSISDPointCapGame(1, 1.2)
	x1_b, x2_b, xi_b = g.play(dstr=strategy_barrier, 
								istr=strategy_barrier)

	plt.figure(figsize=(6.8, 4.8))
	plt.plot(x1_b[:,0], x1_b[:,1], 
				'-o', markevery=70, lw=lw, color='b')
	plt.plot(x2_b[:,0], x2_b[:,1], 
				'-o', markevery=70, lw=lw, color='b')
	plt.plot(xi_b[:,0], xi_b[:,1], 
				'-o', markevery=70, lw=lw, color='r')

	plt.plot(xi_b[-1,0], xi_b[-1,1], 'd', lw=lw, color='r', zorder=100)
	plt.plot(x1_b[-1,0], x1_b[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_b[-1,0], x2_b[-1,1], 'o', lw=lw, color='b')

	plt.text(xi_b[0,0], xi_b[0,1]+.2, r'$I$', fontsize=fs*.9)
	plt.text(x1_b[0,0], x1_b[0,1]-.6, r'$D_1$', fontsize=fs*.9)
	plt.text(x2_b[0,0]-.1, x2_b[0,1]-.6, r'$D_2$', fontsize=fs*.9)

	g.reset()
	x1_p, x2_p, xi_p = g.play(dstr=strategy_barrier, 
								istr=strategy_pass)

	plt.plot(x1_p[:,0], x1_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75, label=r'$D, proposed$')
	plt.plot(x2_p[:,0], x2_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75)
	plt.plot(xi_p[:,0], xi_p[:,1], 
				'--o', markevery=70, lw=lw, color='r', alpha=.75, label=r'$I, Strategy (26b)$')
	plt.plot(xi_p[-1,0], xi_p[-1,1], 'd', lw=lw, color='r', zorder=100)
	plt.plot(x1_p[-1,0], x1_p[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_p[-1,0], x2_p[-1,1], 'o', lw=lw, color='b')

	plt.gca().add_patch(plt.Circle((0, -25), 23, ec='b', fc='lightsteelblue', lw=2))
	plt.text(-.6, -3, r'$Target$', fontsize=fs*.9)

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

	plt.gca().legend([lc_opt, lc_sub], ['(16) vs. (17)', '(16) vs. (26b)'], 
				handler_map={type(lc_opt): HandlerDashedLines()},
	          	handlelength=2.5, handleheight=3, 
	          	fontsize=fs*.9)


	plt.xlabel(r'$x (m)$', fontsize=fs)
	plt.ylabel(r'$y (m)$', fontsize=fs)

	plt.gca().tick_params(axis="both", which="major", labelsize=fs)
	plt.gca().tick_params(axis="both", which="minor", labelsize=fs)
	plt.subplots_adjust(bottom=.17, top=0.95, left=.13, right=0.96)
	plt.grid()
	plt.axis('equal')
	plt.xlim((-6, 6))
	plt.ylim((-3, 4))
	plt.show()

############### change invaders trategy (Figure 7) ###############
def compare_istrategy():

	plt.figure(figsize=(6.8, 5.8))

	# first set of strategy
	g = TDSISDPointCapGame(1, 1.2)
	x1_b, x2_b, xi_b = g.play(dstr=strategy_barrier, 
								istr=strategy_barrier)

	plt.plot(x1_b[:,0], x1_b[:,1], 
				'-o', markevery=70, lw=lw, color='b')
	plt.plot(x2_b[:,0], x2_b[:,1], 
				'-o', markevery=70, lw=lw, color='b')
	plt.plot(xi_b[:,0], xi_b[:,1], 
				'-o', markevery=70, lw=lw, color='r')

	plt.plot(x1_b[-1,0], x1_b[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_b[-1,0], x2_b[-1,1], 'o', lw=lw, color='b')
	plt.plot(xi_b[-1,0], xi_b[-1,1], 'd', lw=lw, color='r', zorder=100)
	
	plt.text(xi_b[0,0], xi_b[0,1]+.2, r'$I$', fontsize=fs*.9)
	plt.text(x1_b[0,0]-.55, x1_b[0,1]-.6, r'$D_1$', fontsize=fs*.9)
	plt.text(x2_b[0,0]-.1, x2_b[0,1]-.6, r'$D_2$', fontsize=fs*.9)

	# second set of strategy
	g.reset()
	x1_p, x2_p, xi_p = g.play(dstr=strategy_pass, 
								istr=strategy_barrier)

	plt.plot(x1_p[:,0], x1_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75)
	plt.plot(x2_p[:,0], x2_p[:,1], 
				'--o', markevery=70, lw=lw, color='b', alpha=.75)
	plt.plot(xi_p[:,0], xi_p[:,1], 
				'--o', markevery=70, lw=lw, color='r', alpha=.75)
	plt.plot(xi_p[-1,0], xi_p[-1,1], 'd', lw=lw, color='r', zorder=100)
	plt.plot(x1_p[-1,0], x1_p[-1,1], 'o', lw=lw, color='b')
	plt.plot(x2_p[-1,0], x2_p[-1,1], 'o', lw=lw, color='b')


	# labels and legends
	plt.gca().add_patch(plt.Circle((0, -25), 23, ec='b', fc='lightsteelblue', lw=2))
	plt.text(-.6, -4, r'$Target$', fontsize=fs*.9)

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

compare_istrategy()
compare_dstrategy()
