# originally in base_2dsihg.py

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
            
    def ebarrier_fit_weight(self, order=4, force_recal=False):

        def fit_gmm(kgmm):
            '''
            returns a regression for L=reg(theta) under given gamma
            input:      kgmm: [0, 1] the location of gamma in [gmm_min, gmm_max]
            output:     reg: regression model, contains weight
                        gmm: gamma
            '''
            
            # generate data for the regression
            gmm_min = self.lb/2
            gmm_max = pi/2
            gmm = gmm_min + kgmm*(gmm_max - gmm_min)
            phi = self.get_max_phi(gmm=gmm)

            xyL, _ = self.tdsi_ebarrier(phi, 0, gmm, n=20, frame='xyL')
            x, y, L = xyL[4:-1,0], xyL[4:-1,1], xyL[4:-1,2]
            # print(x, y, L)
            t = np.array([atan2(yy, xx+LL) for (xx, yy, LL) in zip(x, y, L)])

            # get features
            tt = np.array([p**2 for p in t])
            ttt = np.array([p**3 for p in t])

            # the regression is: w0 + w1*t + w2*t**2 + w3*t**3 = L
            data = np.vstack((t, tt, ttt)).transpose()
            label = L

            # do the regression
            reg = linear_model.LinearRegression()
            reg.fit(data, label)

            # save data
            np.save('./bases/hgdata/gmm%.4f.npy'%gmm, np.vstack((t, L)))

            # get the weights
            w0 = reg.intercept_
            ws = list(reg.coef_)

            return [w0] + ws, gmm

        def fit_weight(w_, g_, order=order):
            '''
            for the weight computed in fig_gmm, returns a regression for weight=reg(gmm)
            input:      w_: weights
                        g_: gamma
            '''

            gpower = [np.array([d**o for d in g_]) for o in range(1, order)]

            data = np.vstack(gpower).transpose()
            label = w_

            reg = linear_model.LinearRegression()
            reg.fit(data, label)

            def get_w(g_, order=order):
                gpower = [g_**o for o in range(1, order)]
                return reg.predict([gpower]).squeeze()

            w0 = reg.intercept_
            ws = reg.coef_

            # return get_w, [w0]+list(ws)
            return [w0]+list(ws)


        if (not os.path.isfile('./bases/hgdata/weights.npy')) or force_recal:

            print('calculating weights')

            # get data for weights and gmm
            w, gmm = fit_gmm(.1)
            nw = len(w) # get the number of weights in fitting L(theta)
            ws, gmms = [[] for _ in range(nw)], []
            
            # for each gmm, fit L(theta), get the weights
            for k in np.linspace(.01, .99, 20):
                w, gmm = fit_gmm(k)

                gmms.append(gmm) 
                for ws_, w_ in zip(ws, w):
                    ws_.append(w_)
                         
            gmms = np.asarray(gmms)
            np.save('./bases/hgdata/gmms.npy', gmms)

            for i in range(len(ws)):
                ws[i] = np.asarray(ws[i])
                np.save('./bases/hgdata/w'+str(i)+'s.npy', ws[i])

            weights = np.array([fit_weight(w, gmms) for w in ws])
            np.save('./bases/hgdata/weights.npy', weights)

        else:
            print('loading weights')

            # TODO: name the weights with game parameters
            weights = np.load('./bases/hgdata/weights.npy')

        return weights

    def reg_thetagmm2L(self, tht, gmm):
        '''
        [1, t, t^2][ [w00, w01, ... w0k] [1
                     [w10, w11, ... w1k]  g
                     ...                  ...
                     [wn0, wn1, ... wnk]] g^k ] = L(theta, g)
        '''

        # torder, gorder = self.weights.shape
        g_ = np.array([gmm**i for i in range(self.gorder)])
        t_ = np.array([tht**i for i in range(self.torder)])        

        return t_.dot(self.weights.dot(g_))        

    def whichss(self, x, y, z):

        '''
        the semipermeable surfaces are represented as
        [x    [ rcos(theta) - L         [-rsin(theta) - L'
         y  =   rsin(theta)         + k   rcos(theta)
         z]         L           ]              L'         ]
        which is equivalent to:
        x + z = rcos(theta) - krsin(theta)
            y = rsin(theta) + krcos(theta)

        On the one hand,
        (x + z)^2 + y^2 = r^2(1 + k^2), 
        which can be used to solve k

        On the other hand,
        (x + z)cos(theta) +       ysin(theta) = r
              ycos(theta) - (x + z)sin(theta) = kr
        which can be used to solve theta

        '''

        den = (x + z)**2 + y**2
        k = sqrt(den/self.l**2 - 1)

        ctheta = ((x + z) + k*y)*self.l/den
        stheta = (y - (x + z)*k)*self.l/den

        # print(k, ctheta, stheta)

        theta = atan2(stheta, ctheta)
        # # print(theta)

        # the coefficient of theta in L(theta) + kL'(theta), 
        # for 1, theta, theta^2, ...
        tcoef = np.array([theta**i + k*i*theta**(i-1)
                            for i in range(self.torder)])
        gcoef = tcoef.dot(self.weights)

        # gcoef[0] + gcoef[1]*gmm + gcoef[2]*gmm^2, ... = z
        gcoef[0] = gcoef[0] - z

        roots = np.roots(gcoef[::-1])
        roots = [r.real for r in roots if abs(r.imag)<1e-6]
        roots = np.array(roots).squeeze()

        return roots


# originally in base_2dsihg.py __main__
    # for p, t, xyL_ in zip(phi, tau, xyL):
    #     print(p, t, xyL_[-1])

    # gmm = g.whichss(x, y, L)
    # theta = atan2(y, x+L)
    # L = g.reg_thetagmm2L(theta, gmm)
    # print(L)

    # ####### verify the approximation for the weights of L(theta) as functions of gamma
    # order = g.weights.shape[-1]

    # gmms = np.load('./bases/hgdata/gmms.npy')
    # ws = [np.load('./bases/hgdata/w'+str(i)+'s.npy') for i in range(4)]

    # ws_ = [[] for _ in range(4)]
    # for gm in gmms:
    #     g_ = np.array([gm**i for i in range(order)])
    #     [w_.append(w__) for w_, w__ in zip(ws_, g.weights.dot(g_))]

    # for w, w_ in zip(ws, ws_):
    #     plt.plot(gmms, w, 'x')
    #     plt.plot(gmms, w_, '--')
    # plt.show()
    
    # ######3# verify the approximation for L(theta) under given gmm
    # for gmm in [0.5955, 0.9004, 1.2561, 1.5609]:
    #     # gmm = 1.1545
    #     data = np.load('./bases/hgdata/gmm'+str(gmm)+'.npy')
    #     theta = data[0]
    #     L = data[1]

    #     plt.plot(theta, L, 'x')
    #     plt.plot(theta, [g.reg_thetagmm2L(t_, gmm) for t_ in theta], '-')
    # plt.show()        