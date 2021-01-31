import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

import cv2

from util import norm
from base_2dsi import TDSISDFixedPhiParam

c = ['k', 'b', 'r', 'm', 'y', 'g', 'c']

g = TDSISDFixedPhiParam(1, 1.4)

L = 10
tmin, tmax = g.get_trange_fromL(L)
tstar = g.barrier_e_tmin(L)
# # print('trange: ', tmin, tmax, tstar)
dt = tmax - tstar
# # tstar = g.barrier_e_tmin(L)


canvas = np.zeros([300, 800, 3],np.uint8)
canvas[:,:] = [255, 255, 255]

for t in np.linspace(tstar+0.0*dt, tmax-0.5*dt, 100):
	xi, xc, v1, v2, vi = g.barrier(L, t)
	xi = xi*50
	vi_ = (vi+g.a*np.array([1, 1]))*125/g.a

	for x, v in zip(xi, vi_):
		# color = (t*20, v[0], v[1]) 
		color = (v[1], v[0], t*40) 
		# print(color)
		i = 250-int(x[1])
		j = int(x[0])
		# print(canvas[i,j]==[0, 0, 0])
		if all(canvas[i,j] == [255, 255, 255]):
			canvas[250-int(x[1]), int(x[0])] = color

	# plt.plot(xi[:,0], xi[:,1], color=c[i%7])
	# plt.plot(xc[:,0], xc[:,1], color=c[i%7])

# plt.grid()
# plt.axis('equal')
# plt.show()


# for cnt, t in zip(cnts, ts):
# 	cv2.drawContours(drawing,[cnt],0,(int(t*100)%255,255,255),1)
# canvas[:, 1:100] = [100, 180, 250]

cv2.imshow('output',canvas)
cv2.waitKey(0)