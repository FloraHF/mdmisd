import numpy as np
from math import pi, sin, cos, tan, acos, asin, atan2, sqrt
from mpmath import ellipe, ellipf
from scipy.optimize import minimize_scalar

from util import norm, dot, cross

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
        return (L - 2*self.l)/self.vd

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

    def sdsi_barrier_e(self, phi, tau):
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

        x1 = x[:2]
        xi = x[2:]
        # print(norm(x1 - xi))

        return x, t

    def tdsi_barrier_e(self, phi, tau, gmm):

        assert 0 <= gmm <= pi

        x, t = self.sdsi_barrier_e(phi, tau)
        d = t*self.vd + self.l

        x_D2 = d*sin(2*gmm - self.lb)
        y_D2 = d*cos(2*gmm - self.lb)

        x = np.array([x[0], x[1], x_D2, y_D2, x[2], x[3]])
        
        return x, t

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
            x, _ = self.tdsi_barrier_e(phi, 0, gmm)
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

        return np.array([x, y])

    def isoc_e(self, L, t, n=10):

        # maximum time for phase II
        t_ub = min(self.t2_max, t) 

        # time spent in phase II
        t2s = np.linspace(0, t_ub, n)

        xis = []

        for t2 in t2s:

            tau = t - t2
            phi = self.phi_from_t(t2)
            print(phi - self.lb)

            x, _ = self.sdsi_barrier_e(phi, tau)

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

        return np.asarray(xis)

    def isoc_n(self, L, t, n=25):
        
        d = t*self.vd + self.l
        gmm = asin(L/2/d)
        tht = gmm - self.lb/2

        cx = 0
        cy = -d*cos(gmm)

        g = np.linspace(pi/2 + gmm - self.lb/2, pi/2, n)
        
        xi = (cx + self.vi/self.vd*d*np.cos(g)).reshape(-1, 1)
        yi = (cy + self.vi/self.vd*d*np.sin(g)).reshape(-1, 1)

        return np.concatenate((xi, yi), 1)

if __name__ == '__main__':
    
    g = TDSISDHagdorn(1, 1, 1.2)
    tmin = g.tmin(2)
    tmax = g.tmax(2)
    t = (tmin + tmax)/2
    print(g.isoc_e(2, t))