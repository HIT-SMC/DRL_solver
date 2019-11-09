# -*- coding:utf-8　-*-
"""
building the ODE environment of Lorenz Equation
Author: Shiyin WEI
Date: Apr. 7, 2018
"""

import numpy as np

class Lorenz(object):
    def __init__(self,pars = [10., 8./3, 28.]):
        self.state = np.zeros([1,7])  # variables: x,y,z, xt,yt,zt
        self.sigma, self.beta, self.r = pars
    def _reset(self,init=[0., 2., 0.]):
        t = 0.
        x, y, z = init
        xt = self.sigma*(y-x)
        yt = self.r*x - y - x*z
        zt = -self.beta*z + x*y
        self.state = np.hstack([x, y, z, xt, yt, zt, t])
        return self.state
    def _step(self,action):
        # self.state = action
        x, y, z, xt, yt, zt = action
        r = np.square(xt - self.sigma*(y - z)) + \
            np.square(yt - self.r*x + y + x*z) + \
            np.square(zt + self.beta*z - x*y)
        return r/1e3


