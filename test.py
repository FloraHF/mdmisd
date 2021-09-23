import numpy as np
from math import pi, cos, sin
from sklearn import linear_model

import matplotlib.pyplot as plt

from base_2dsihg import TDSISDHagdorn

g = TDSISDHagdorn(1, 1, 1.2)

phi_min = g.lb
gmm_min = phi_min/2
gmm_max = pi/2
gmm = gmm_min + .01*(gmm_max - gmm_min)
ax = plt.figure().add_subplot(projection='3d')


x, t = g.tdsi_nbarrier(5, gmm, .0, frame='xyL')    
# ax.plot(x[:,0], x[:,1], x[:,2], 'r-o')

X = x[:,0:2]
Y = x[:, 2]

for k in [.1, .2, .3, .5, .7]:
    phi = phi_min + k*(g.get_max_phi(gmm=gmm) - phi_min)
    print(phi, gmm)
    x, t = g.tdsi_ebarrier(phi, 7, gmm, n=20, frame='xyL')
    ax.plot(x[:,0], x[:,1], x[:,2], 'r-o', markersize=2)

    X = np.concatenate((X, x[:,:2]))
    Y = np.concatenate((Y, x[:, 2]))

# print(X[0])
# print(np.asarray(Y))

reg = linear_model.LinearRegression()
model = reg.fit(X, Y)
r = model.score(X, Y)

w0 = reg.intercept_
w1, w2 = reg.coef_

print(w0, w1, w2)
print(w0 + w1*1 + w2*2)
print(model.predict([[1, 2]]))

X = np.arange(-15, 1, 0.25)
Y = np.arange(-.5, 2, 0.25)
X, Y = np.meshgrid(X, Y)

XY = np.array([X.flatten(), Y.flatten()]).T
Z = model.predict(XY).reshape(X.shape)

# Z = np.zeros(X.shape)

# for i, (xx, yy) in enumerate(zip(X, Y)):
# 	for j, (x, y) in enumerate(zip(xx, yy)):
# 		Z[i,j] = model.predict(np.array([[x, y]])).squeeze()

ax.plot_surface(X, Y, Z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('L')
plt.show()