import numpy as np
from math import pi, cos, sin
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, constraints
from tensorflow.keras import backend as K
# from tensorflow.keras.layers.experimental import preprocessing

from barrier import read_barrier, BarrierNormlizer, c
from base_2dsi import TDSISDFixedPhiParam

class NNModel(object):
	"""docstring for NNmodel"""
	def __init__(self, feature=['L','t', 'x'],
						label='y'):

		self.feature = feature
		self.indim = len(feature)
		self.label = label
		self.mname = 'models/'+'_'.join([''.join(feature), label])+'.h5'
		self.normalize = True

	def build_nn(self, normalizer):

		model = tf.keras.Sequential([
			normalizer,
			layers.Dense(32, activation='tanh', 
						kernel_constraint=constraints.max_norm(1)),
			layers.Dropout(.3),
			layers.Dense(32, activation='tanh', 
						kernel_constraint=constraints.max_norm(1)),
			# layers.Dense(64, activation='tanh'),
			layers.Dense(1)
			])

		model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
					  loss='mse')

		return model

	def train_nnreg(self):		
		data, label = read_barrier(feature=self.feature, 
									label=self.label,
									normalize=self.normalize)
		normalizer = preprocessing.Normalization(
							input_shape=[self.indim,])
		normalizer.adapt(data)

		model = self.build_nn(normalizer)

		history = model.fit(data, label,
					epochs=500,
					batch_size=64,
					verbose=1,
					validation_split=0.3,
					shuffle=True)

		# print(history.history.keys())
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.grid()
		plt.show()

		model.save(self.mname)

	def verify_model(self):
		model = tf.keras.models.load_model(self.mname)
		model.summary()

		L = 10
		g = TDSISDFixedPhiParam(1, 1.4)
		n = BarrierNormlizer(g)

		tmin, tmax = g.get_trange_fromL(L)
		tstar = g.barrier_e_tmin(L)
		ts = [t for t in np.linspace(tstar, tmax, 3)]
		xs = [x for x in np.linspace(0, L/2, 50)]

		for t in ts:
			xs_in_ = np.array([[L, n.t_(t, L), n.x_(x, L)] 
			 				  for x in xs])
			ys_ = model.predict(xs_in_)
			ys = n.y(ys_, L, t)

			plt.plot(xs, ys)

def verify_bdata():
	xx, yy = read_barrier(feature=['x','y', 'L'], label='t')

	k, s = 0, 0
	x_ = []
	y_ = yy[0]
	L_ = xx[0,2]
	for x, y in zip(xx, yy):
		if y == y_:
			x_.append(x)
		else:
			x__ = np.asarray(x_)
			plt.plot(x__[:,0], x__[:,1], '.--', color=c[s%7])

			tht = np.linspace(0, 2*pi, 50)
			xr = L_/2 + np.cos(tht)
			yr = 0 + np.sin(tht)
			plt.plot(xr, yr, '--', color=c[s%7])

			x_ = []
			y_ = y
			k += 1
		if x[2] != L_:
			L_ = x[2]
			s += 1
	
	plt.grid()
	plt.axis('equal')
	plt.show()

verify_bdata()

# m = NNModel()
# # m.train_nnreg()
# m.verify_model()

# plt.axis('equal')
# plt.show()