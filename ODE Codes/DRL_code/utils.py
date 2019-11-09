# -*- coding:utf-8 -*--
# Copyright @ Shiyin WEI
# corresponding to weishiyinhit@gmail.com
# Date: Jan. 10, 2018
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import random

class solver():  # 求解器参数
    def __init__(self, batch_size, timestep_size, learning_rate, feature_size, hidden_size, layer_num, delta_t):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.timestep_size = timestep_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.delta_t = delta_t


class gmotion(object):
    ''' read ground acceleration record
        eith 1-d numpy array or csv file'''

    def __init__(self, gmotion_file, dt_gmotion, period=None, psa=None):

        if isinstance(gmotion_file, np.ndarray):
            try:
                assert(gmotion_file.ndim==1)
                self.ground = gmotion_file
            except AssertionError:
                self.ground = gmotion_file.flatten()
        else:
            self.ground = np.loadtxt(gmotion_file)

        self.dt = dt_gmotion
        self.npts = len(self.ground)
        self.ts = np.linspace(0.0, (self.npts-1)*dt_gmotion, self.npts)

        self.period = period
        self.psa = psa


class MDOFmodel(object):

    def __init__(self, mass, damp, stiff, degree):
        self. degree, self.mass, self.damp, self.stiff = degree, mass, damp, stiff

    def hysteresis(self, A, alpha, beta, gamma, n):
        self.A, self.alpha, self.beta, self.gamma, self.n = A, alpha, beta, gamma, n



class model_bilinear(object):
    '''SDOF model definition
    (dy, fy): yield point
    (alpha): k2/k1
    damp_ratio: damping ratio'''

    def __init__(self, mass, period, fy, alpha, damp_ratio, flag_newmark):
        self.mass = mass
        self.period = period # elastic period
        self.fy = fy # yield point
        self.alpha = alpha # k2 = alpha*k1
        self.damp_ratio = damp_ratio # damping ratio zi

        self.omega = 2.0*np.pi/self.period
        self.stiff = mass*self.omega**2.0
        self.dy = fy/self.stiff
        self.damp = 2.0*damp_ratio*self.omega*self.mass
        self.stiff_hat = None
        self.gamma = 1./2
        self.flag_newmark = flag_newmark
        self.beta = 1./4 if flag_newmark=='AA' else 1./6

    def determine_stiff(self, dres_current, force_current, dis_current):

        fsmax = self.fy + self.alpha*self.stiff*(dis_current - self.dy)
        fsmin = -self.fy + self.alpha*self.stiff*(dis_current + self.dy)

        if ((dres_current > 0) & (force_current >= fsmax)) | (
            (dres_current < 0) & (force_current <= fsmin)):
            self.stiff = self.alpha*self.fy/self.dy
        else:
            self.stiff = self.mass*self.omega**2.0

    def pushover(self, dis):
        '''elliptical_pushover
            input: dis (array or scalar)
            output: force
        '''
        if isinstance(dis, list):
            dis = np.array(dis)
        elif isinstance(dis*1.0, float):
            dis = np.array([dis*1.0])

        force = np.zeros_like(dis)

        tf = np.abs(dis) <= self.dy
        force[tf] = self.fy/self.dy*dis[tf]

        tf = np.abs(dis) > self.dy
        force[tf] = np.sign(dis[tf])*(
            self.fy + self.alpha*self.stiff*(np.abs(dis[tf])-self.dy))

        return force

    def hysteresis(self, force_current, dis_current, dis_new):
        ''' ellipitical_hysteresis
        input: fs0:
                d: 2x1
                backbone
        output: a0, hysteresis
        '''

        fsmax = self.fy + self.alpha*self.stiff*(dis_new - self.dy)
        fsmin = -self.fy + self.alpha*self.stiff*(dis_new + self.dy)

        force_new = force_current + self.stiff*(dis_new - dis_current)

        if force_new > fsmax:
            force_new = fsmax

        if force_new < fsmin:
            force_new = fsmin

        return force_new


def newmark_parameters(model, dt_comp, flag_newmark):

    # Newmark parameters definition
    # -----------------------------
    const_gamma = 0.5
    if flag_newmark == 'LA':
        const_beta = 1.0/6.0   # linear acceleration (stable if Dt/T<=0.551)
        assert dt_comp/model.period <= 0.551, "Newmark method is unstable"
    elif flag_newmark == 'AA':
        const_beta = 0.25   # average acceleration (unconditionally stable)

    const_a = (1.0/(const_beta*dt_comp)*model.mass + (
        const_gamma/const_beta*model.damp))
    const_b = 1.0/(2.0*const_beta)*model.mass + dt_comp*(
        const_gamma/(2.0*const_beta)-1.0)*model.damp

    return (const_a, const_b, const_beta, const_gamma)

def external_force(grecord, dt_comp):

    tseries = np.arange(0.0, grecord.ts[-1], dt_comp)
    flin = interp1d(grecord.ts, grecord.ground)
    eforce = flin(tseries)

    return (tseries, eforce)

def linear_response_sdof(model, grecord, grecord_factor, dt_comp, \
    flag_newmark='AA'):

    '''
    compute response of SDOF model
    input: model (class)
    input: grecord (class)
    output: spectra values
    '''

    # read earthquake force
    # grecord factor should include -1 if it's ground acceleration
    if np.isclose(grecord.dt, dt_comp):
        eforce = grecord_factor * grecord.ground
        ts_index = grecord.ts
    else:
        (ts_index, eforce) = external_force(grecord, dt_comp)
        eforce *= grecord_factor

    npts = len(eforce)

    (const_a, const_b, const_beta, const_gamma) = newmark_parameters(
        model, dt_comp, flag_newmark)

    dis = np.zeros((npts))
    vel = np.zeros((npts))
    acc = np.zeros((npts))

    # Initial calculations
    # --------------------
    acc[0] = (eforce[0] - model.damp*vel[0] - model.stiff*dis[0])

    model.stiff_hat = model.stiff + const_a/dt_comp

    # Time stepping (Table 5.7.2)
    # -------------
    for i in range(npts-1):

        # 2.1
        delta_phat = eforce[i+1]-eforce[i] + const_a*vel[i] + const_b*acc[i]

        # 2.2 determine the tangent stiffness ki
        delta_dis = delta_phat/model.stiff_hat

        # 2.3
        delta_vel = const_gamma/(const_beta*dt_comp)*delta_dis -(
            const_gamma/const_beta*vel[i]) + dt_comp*(1.0-const_gamma/(
                2.0*const_beta))*acc[i]

        # 2.4
        delta_acc = 1.0/(const_beta*dt_comp**2)*delta_dis - 1.0/(
            const_beta*dt_comp)*vel[i] - 1.0/(2.0*const_beta)*acc[i]

        # 2.7
        dis[i+1] = dis[i] + delta_dis
        vel[i+1] = vel[i] + delta_vel
        acc[i+1] = acc[i] + delta_acc

    # Spectral values
    # ---------------
    tacc = acc - eforce/model.mass

    response = pd.DataFrame(np.vstack((dis, vel, acc, tacc)).T, \
        index=ts_index, columns=['dis', 'vel', 'acc', 'tacc'])

    return response, eforce

def nonlinear_response_sdof(model, grecord, grecord_factor, dt_comp, \
    flag_newmark='AA'):

    '''
    compute response of SDOF model with a hysteresis
    input: model (class)
    input: grecord (class)
    output: spectra values
    '''

    # read earthquake force
    # grecord factor should include -1 if it's ground acceleration
    if np.isclose(grecord.dt, dt_comp):
        eforce = grecord_factor * grecord.ground
        ts_index = grecord.ts
    else:
        (ts_index, eforce) = external_force(grecord, dt_comp)
        eforce *= grecord_factor
    npts = len(eforce)

    (const_a, const_b, const_beta, const_gamma) = newmark_parameters(
        model, dt_comp, flag_newmark)

    dis = np.zeros((npts))
    vel = np.zeros((npts))
    acc = np.zeros((npts))
    force = np.zeros((npts))

    # Initial calculations
    # --------------------
    acc[0] = (eforce[0] - model.damp*vel[0] - force[0])

    if hasattr(model, 'ref_d0'):
        model.ref_d0 = 0.0
        model.Iunloading = 0

    # Time stepping (Table 5.7.2)
    # -------------
    for i in range(npts-1):
        #print "Time: %s" %i
        # 2.1

        delta_phat = eforce[i+1]-eforce[i] + const_a*vel[i] + const_b*acc[i]

        # determine ki
        model.determine_stiff(delta_phat, force[i], dis[i])

        model.stiff_hat = model.stiff + const_a/dt_comp

        # 2.2 determine the tangent stiffness ki
        (delta_dis, force[i+1]) = modified_newton_raphson_method(dis[i], \
            force[i], delta_phat, model)

        # 2.5
        delta_vel = const_gamma/(const_beta*dt_comp)*delta_dis -(
            const_gamma/const_beta*vel[i]) + dt_comp*(
                1.0-const_gamma/(2.0*const_beta))*acc[i]

        # 2.7
        dis[i+1] = dis[i] + delta_dis
        vel[i+1] = vel[i] + delta_vel
        acc[i+1] = (eforce[i+1] - model.damp*vel[i+1] - force[i+1])/model.mass

    tacc = acc - eforce/model.mass

    response = pd.DataFrame(np.vstack((dis, vel, acc, tacc, force)).T, \
        index=ts_index, columns=['dis', 'vel', 'acc', 'tacc', 'force'])

    return response, eforce

def modified_newton_raphson_method(dis_current, force_current, dres_current, \
    model):

    # Table 5.7.1. from Chopra Book
    const_tol = 1e-8
    max_iter = 20
    incr_ratio = 1.0
    j = 0

    delta_u = np.zeros((max_iter, 1))
    delta_u[0] = dres_current/model.stiff_hat

    while (incr_ratio > const_tol) & (j < max_iter):

        #print "deltaD: %s, incr_ratio: %s" %(delta_u[j], incr_ratio)
        dis_new = dis_current + delta_u[j]

        # determine force(j)
        force_new = model.hysteresis(force_current, dis_current, \
            dis_new)

        dres_new = dres_current - (force_new - force_current + (
            model.stiff_hat-model.stiff)*delta_u[j])

        delta_u[j+1] = dres_new/model.stiff_hat # 2.1

        incr_ratio = delta_u[j+1]/np.sum(delta_u)

        j += 1 # increase index

        # update dis, force
        dis_current = dis_new
        force_current = force_new
        dres_current = dres_new

    return (np.sum(delta_u), force_new)

class experience_buffer():
    def __init__(self, buffer_size):  # buffer size = 50000
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 4])