import numpy as np
from math import sqrt

def dot(x, y):
	s = 0
	for xx, yy in zip(x, y):
		s += xx*yy
	return s

def cross(x, y):
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