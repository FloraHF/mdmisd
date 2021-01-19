import numpy as np
from math import sqrt

def dot(x, y):
	s = 0
	for xx, yy in zip(x, y):
		s += xx*yy
	return s

def norm(x):
	return sqrt(dot(x, x))

def dist(x, y):
	return norm(x-y)