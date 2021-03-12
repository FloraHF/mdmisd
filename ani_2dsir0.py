from game_2dsir0 import TDSISDPointCapGame
from base_2dsir0 import strategy_barrier, strategy_pass, strategy_default

import matplotlib.pyplot as plt
import csv
from scipy import stats


# first set of strategy
g = TDSISDPointCapGame(1, 1.2, dt=0.05)
x1, x2, xi = g.play(dstr=strategy_barrier, 
					istr=strategy_barrier, 
					render=True,
					record=True)

with open('time.csv', 'r') as f:
	reader = csv.reader(f)
	tts, tmins, ts, tmaxs, tpct = [], [], [], [], []
	for i, row in enumerate(reader):
		tts.append(i*0.05)
		tmins.append(float(row[0]))
		ts.append(float(row[1]))
		tmaxs.append(float(row[2]))
		tpct.append((ts[-1] - tmins[-1])/(tmaxs[-1] - tmins[-1]))
		# tmin_rate
		# tau_rate
		# tmax_rate
# for t1, t2 in zip(ts[:-1], ts[1:]):
# 	print(t1-t2)
# k, b, r, p, std = stats.linregress(tts, ts)
# print(std)

plt.plot(tts, tpct)
# for tt, tpct in zip(tts, tpct):
# 	print(tt, tpct)
# plt.plot(tts, tpct, label='tmin')
# plt.plot(tts, tmaxs, label='tmax')
# plt.plot(tts, ts, label='t')

plt.grid()
plt.legend()
plt.show()
