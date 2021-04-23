import numpy as np
from math import sqrt, log


#####################################################
#                  	basic math						|
#####################################################
def dot(x, y):
	s = 0
	for xx, yy in zip(x, y):
		s += xx*yy
	return s

def cross(x, y):
	if len(x) == 2:
		x = np.array([x[0], x[1], 0])
	if len(y) == 2:
		y = np.array([y[0], y[1], 0])
	assert len(x) == 3
	assert len(y) == 3
	e1 = x[1]*y[2] - x[2]*y[1]
	e2 = x[2]*y[0] - x[0]*y[2]
	e3 = x[0]*y[1] - x[1]*y[0]
	return np.array([e1, e2, e3])

def norm(x):
	return sqrt(dot(x, x))

def dist(x, y):
	return norm(x-y)

def logb(x, b=10):
	return log(x)/log(b)


#####################################################
#                  	coordinates						|
#####################################################

# if __name__ == '__main__':
# 	tmin = 1
# 	tmax = 4
# 	b = 100
# 	par = np.linspace(logb(tmin, b=b), logb(tmax, b=b), 5)
# 	print(par)
# 	print([b**p for p in par])


#####################################################
#                  		plots						|
#####################################################
def plot_cap_range(ax, x, r, n=50):
	t = np.linspace(0, 2*np.pi, n)
	x_ = x[0] + r*np.cos(t)
	y_ = x[1] + r*np.sin(t)
	ax.plot(x_, y_, color='b')