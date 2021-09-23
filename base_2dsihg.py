import numpy as np
from math import pi, sin, cos, tan, acos, asin, atan2, sqrt
from mpmath import ellipe, ellipf
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from util import norm, dot, cross, plot_cap_range

class TDSISDHagdorn(object):
    """docstring for TDSISDHagdorn"""
    def __init__(self, r, vd, vi):
        
        self.vd = vd
        self.vi = vi
        self.w = vi/vd
        self.l = r

        self.lb = 2*acos(1/self.w)
        self.ct = self._t_from_phi(self.lb)

        self.phi_max = self.get_max_phi()
        # # print(phi_max)
        self.t2_max = self.t_from_phi(self.phi_max)

    def tmax(self, L):
        return (L/2/sin(self.lb/2) - self.l)/self.vd

    def tmin(self, L):
        return (L/2 - self.l)/self.vd

    def _t_from_phi(self, phi):

        l = self.l/self.vd

        A = l/(self.w**2 - 1)
        c = cos(phi/2)

        E = ellipe(asin(self.w*c), 1/self.w)
        F = ellipf(asin(self.w*c), 1/self.w)

        t = 2*l*F/self.w - 3*self.w*E*A - 3*self.w*c*A 

        return float(t)

    def t_from_phi(self, phi):

        return self._t_from_phi(phi) - self.ct

    def phi_from_t(self, t):

        # assert t < self.t2_max

        def t_mismatch(phi):
            dt = t - self.t_from_phi(phi)
            return dt**2

        res = minimize_scalar(t_mismatch, bounds=[self.lb, 2*pi - self.lb], method='bounded')

        return res.x

    def sdsi_ebarrier_point(self, phi, tau):
        """ trajectory under opitmal play, for subgame between invader and the closer defender. 
            reference frame: physical 6D frame, y is along ID_2 upon capture
            ----------------------------------------------------------------------------
            Input:
                phi: phi_1 in [lb, ub]                                      float
                tau: time spent on the straight line trajectory             float
                lb:  lower bound of phi.                                    float
                     lb=LB: at barrier,
                     lb>LB: other SS
            Output:
                x: 4D trajectory, order: D1, I                              np.array
            ----------------------------------------------------------------------------
        """

        l = self.l/self.vd
        w = self.w
        # LB = self.lb
        
        def consts(phi):
            s, c = sin(phi/2), cos(phi/2)
            A, B = l/(w**2 - 1), sqrt(1 - w**2*c**2)
            return s, c, A, B

        # phase II trajectory
        def xtau(phi):

            out = np.zeros((4,))

            s, c, A, B = consts(phi)

            E = float(ellipe(asin(w*c), 1/w))
            F = float(ellipf(asin(w*c), 1/w))

            out[0] = - 2*A*(s**2*B + w*s**3)                                        # x_D1
            out[1] = A*(w*E + 3*w*c + 2*s*c*B - 2*w*c**3)                           # y_D1
            out[2] = A*(B*(2*w**2*c**2 - w**2 - 1) - 2*w**3*s**3 + 2*w*(w**2-1)*s)  # x_I
            out[3] = A*(w*E + 2*w**2*s*c*B - 2*w**3*c**3 + w*c*(2+w**2))            # y_I
            t = 2*l*F/w - 3*w*E*A - 3*w*c*A  

            return out, t

        s, c, A, B = consts(phi)

        # phase I trajectory plus initial location due to capture range
        xtemp = np.zeros((4,))
        xtemp[0] = -l*sin(self.lb) - 2*s*c*tau                              # x_D1
        xtemp[1] =  l*cos(self.lb) + (2*c**2 - 1)*tau                       # y_D1
        xtemp[2] =            - (w**2*s*c + w*c*B)*tau                      # x_I
        xtemp[3] =            + (w**2*c**2 - w*s*B)*tau                     # y_I

        xf, tf = xtau(phi)
        x0, t0 = xtau(self.lb)

        dt = max(tf - t0, 0)
        dx = xf - x0
        
        x = dx + xtemp
        t = dt + tau
        
        x = x*self.vd

        # x1 = x[:2]
        # xi = x[2:]
        # print(norm(x1 - xi))

        return x, t

    def tdsi_ebarrier_point(self, phi, tau, gmm, frame='sol'):

        ''' input: frame: reference frame 
                          xyL: 
                          sol: the frame for solving the trajectory  
        '''


        assert 0 <= gmm <= pi
        assert (frame == 'xyL' or frame == 'sol')

        x, t = self.sdsi_ebarrier_point(phi, tau)
        d = t*self.vd + self.l

        x_D2 = d*sin(2*gmm - self.lb)
        y_D2 = d*cos(2*gmm - self.lb)

        x = np.array([x[0], x[1], x_D2, y_D2, x[2], x[3]])

        if frame == 'xyL': x = self.to_xyL(x)
        
        return x, t

    def to_xyL(self, x):

        x1 = x[:2]
        x2 = x[2:4]
        xi = x[4:]

        vd = x2 - x1        # vector from D1 to D2
        L = norm(vd)
        ed = vd/L
        ed = np.array([ed[0], ed[1], 0])
        ex = np.array([1, 0, 0])

        sin_a = cross(ex, ed)[-1]
        cos_a = dot(ex, ed)

        # print(sin_a, cos_a)
        C = np.array([[cos_a, sin_a],
                     [-sin_a, cos_a]])

        xm_ = 0.5*(x1 + x2)  # mid point of |D1 D2|

        # translate locations
        xi_ = C.dot(xi - xm_)
        x = np.array([xi_[0], xi_[1], L/2])

        return x


    def tdsi_ebarrier(self, phi, tau, gmm, n=20, frame='sol'):
        
        assert 0 <= gmm <= pi
        assert tau >= 0

        phi_max = self.get_max_phi(gmm=gmm)
        assert self.lb <= phi <= phi_max
        
        xs, ts = [], []        
        for p in np.linspace(self.lb, phi, n):
            x, t = self.tdsi_ebarrier_point(p, 0, gmm, frame=frame)
            xs.append(x)
            ts.append(t)

        if tau > 0:
            for tau_ in np.linspace(0, tau, 10):
                x_, t = self.tdsi_ebarrier_point(p, tau_, gmm, frame=frame)
                xs.append(x_)
                ts.append(t)

        return np.asarray(xs), np.asarray(t)

    def tdsi_nbarrier(self, t, gmm, dlt, n=10, frame='phy'):

        assert 2*gmm >= self.lb
        assert 0 <= dlt <= gmm - self.lb/2

        xs = []
        for t_ in np.linspace(0, t, n):

            Ld = self.vd*t_ + self.l
            Li = self.vi*t_
            x1 = -Ld*sin(gmm)
            y1 =  Ld*cos(gmm)
            x2 =  Ld*sin(gmm)
            y2 =  Ld*cos(gmm)
            xi = -Li*sin(dlt)
            yi =  Li*cos(dlt)

            x = np.array([x1, y1, x2, y2, xi, yi])

            if frame == 'xyL': x = self.to_xyL(x)

            xs.append(x)

        return np.asarray(xs), t

    def get_max_phi(self, gmm=None):
        """ obtain the maximum phi
            ----------------------------------------------------------------------------
            Input:
                lb:  lower bound of phi.                                            float
                     lb=LB: at barrier,
                     lb>LB: other SS
            Output:
                ub: (=res.x)                                                        float
                    maximum phi is obtained when I, D2, D1 forms a straight line
            ----------------------------------------------------------------------------
        """
        if gmm is None:
            gmm = self.lb/2

        def slope_mismatch(phi):
            x, _ = self.tdsi_ebarrier_point(phi, 0, gmm)
            s_I_D1 = (x[1] - x[5])/(x[0] - x[4])
            s_I_D2 = (x[3] - x[5])/(x[2] - x[4])
            return (s_I_D1 - s_I_D2)**2

        res = minimize_scalar(slope_mismatch, bounds=[pi, 2*pi-self.lb], method='bounded')

        return res.x

    def xd2(self, L, t, xd1):

        # find xd2 = [x, y], the intersection of:
        # C1: x^2 + y^2 = (self.l + self.vd*t)^2
        # C2: (x - xd1)^2 + (y - yd1)^2 = L^2

        d = self.l + self.vd*t

        a = xd1[0]
        b = xd1[1]
        A = a**2 + b**2
        B = A + d**2 - L**2

        D = 4*A*d**2 - B**2

        # there's no solution if 
        # C1 and C2 don't intersect
        if D < 0:
            return None

        x = (a*B - b*sqrt(D))/(2*A)
        y = (B - 2*a*x)/(2*b)

        # print('---------------')
        # print(sqrt(x**2 + y**2), d)
        # print(sqrt((x - a)**2 + (y - b)**2), L)

        return np.array([x, y])

    def loaf_cylindar(self, Lmax, n=10):
        Lmin = self.r*sin(self.lb/2)
        for L in np.linspace(Lmin, Lmax, n):
            tht = np.linspace(0, 2*pi, 50)
            x = self.r*np.cos(tht) + L
            y = self.r*np.sin(tht)
            z = L*np.ones(n)

    def isoc_e(self, L, t, n=30):

        # maximum time for phase II
        assert t >= self.tmin(L)
        t_ub = min(self.t2_max, t) 

        # time spent in phase II
        t2s = np.linspace(0, t_ub, n)

        xis = []

        for t2 in t2s:

            tau = t - t2
            phi = self.phi_from_t(t2)
            # if t2 == 0:
            #     phi = self.lb
            # print('t', t2, 'difference', phi - self.lb)

            x, _ = self.sdsi_ebarrier_point(phi, tau)

            x1 = x[:2]
            xi = x[2:]

            # solve for x2
            x2 = self.xd2(L, t, x1)

            # if there's no solution for x2
            if x2 is None: 
                xis.append(None)

            vd = x2 - x1        # vector from D1 to D2
            ed = vd/norm(vd)
            ed = np.array([ed[0], ed[1], 0])
            ex = np.array([1, 0, 0])

            sin_a = cross(ex, ed)[-1]
            cos_a = dot(ex, ed)
            C = np.array([[cos_a, sin_a],
                         [-sin_a, cos_a]])

            xm_ = 0.5*(x1 + x2)  # mid point of |D1 D2|

            # translate locations
            # xc = C.dot(xc - xm)
            xi_ = C.dot(xi - xm_)
            x1_ = C.dot(x1 - xm_)
            x2_ = C.dot(x2 - xm_)

            xis.append(xi_)
            # print(x1_, x2_)

        return np.asarray(xis)

    def isoc_n(self, L, t, n=10):
        
        print('-----------inside isoc_n----------')
        print(L, t)

        d = t*self.vd + self.l
        gmm = asin(L/2/d)
        # print(gmm)
        tht = gmm - self.lb/2

        cx = 0
        cy = -d*cos(gmm)

        # g = np.linspace(pi/2 + gmm - self.lb/2, pi/2, n)
        g = np.linspace(pi/2 - tht, pi/2)
        
        xi = -(cx + self.vi*t*np.cos(g)).reshape(-1, 1)
        yi = (cy + self.vi*t*np.sin(g)).reshape(-1, 1)

        return np.concatenate((xi, yi), 1)

if __name__ == '__main__':
    
    g = TDSISDHagdorn(1, 1, 1.2)

    # phi = g.lb + .5
    # t_from_phi = g.t_from_phi(phi)
    # phi_from_t = g.phi_from_t(t_from_phi)
    # print(t_from_phi, phi_from_t, phi, abs(phi_from_t-phi))

    # L = 7
    # tmin = g.tmin(L)
    # tmax = g.tmax(L)
    # # t = (tmin + tmax)/2
    # t = tmin

    # t = tmin + .3*(tmax - tmin)
    # xi_e = g.isoc_e(L, t)
    # xi_n = g.isoc_n(L, t)

    # plt.plot(xi_e[:,0], xi_e[:,1])
    # plt.plot(xi_n[:,0], xi_n[:,1])



    # plt.plot([-L/2], [0], 'bo')
    # plt.plot([ L/2], [0], 'bo')
    # plot_cap_range(plt.gca(), np.array([-L/2, 0]), 1)
    # plot_cap_range(plt.gca(), np.array([ L/2, 0]), 1)

    # plt.axis('equal')
    # plt.show()    

    phi_min = g.lb
    gmm_min = phi_min/2
    gmm_max = pi/2
    gmm = gmm_min + .01*(gmm_max - gmm_min)
    ax = plt.figure().add_subplot(projection='3d')

    for k in [.1, .2, .3, .5, .7]:
        phi = phi_min + k*(g.get_max_phi(gmm=gmm) - phi_min)
        print(phi, gmm)
        x, t = g.tdsi_ebarrier(phi, 7, gmm, n=20, frame='xyL')
        ax.plot(x[:,0], x[:,1], x[:,2], 'r-o')
    x, t = g.tdsi_nbarrier(5, gmm, .0, frame='xyL')    
    ax.plot(x[:,0], x[:,1], x[:,2], 'r-o')

    # phi = phi_min + .2*(g.get_max_phi(gmm=gmm) - phi_min)

    # for k in [.1, .3, .5]:
    #     gmm = gmm_min + k*(gmm_max - gmm_min)
    #     # phi = phi_min + .2*(g.get_max_phi(gmm=gmm) - phi_min)
    #     phi = g.get_max_phi(gmm=gmm)
    #     # print(phi, gmm)
    #     x, t = g.tdsi_ebarrier(phi, .1, gmm, n=20, frame='xyL')
    #     print(x[:,2])
    #     ax.plot(x[:,0], x[:,1], x[:,2])

    gmm = gmm_min + .9*(gmm_max - gmm_min)
    for k in [.1, .2, .3, .5, .7]:
        phi = phi_min + k*(g.get_max_phi(gmm=gmm) - phi_min)
        print(phi, gmm)
        x, t = g.tdsi_ebarrier(phi, 7, gmm, n=20, frame='xyL')
        ax.plot(x[:,0], x[:,1], x[:,2], 'b-o')
    x, t = g.tdsi_nbarrier(5, gmm, .0, frame='xyL')    
    ax.plot(x[:,0], x[:,1], x[:,2], 'r-o')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('L')
    plt.show()


    # phi = phi_min + .1*(g.get_max_phi(gmm=gmm) - phi_min)
    # # x, t = g.tdsi_ebarrier(phi, 4, gmm, n=20, frame='sol')
    # x, t = g.tdsi_nbarrier(5, gmm, .1, frame='phy')
    # plt.plot(x[:,0], x[:,1])
    # plt.plot(x[:,2], x[:,3])
    # plt.plot(x[:,4], x[:,5])

    # plt.show()

