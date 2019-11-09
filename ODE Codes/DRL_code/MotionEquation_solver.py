# -*- coding: utf-8 -*--
# PG solver for MDOF motion equation under El_centro earthquake
# Bouc-wen parameters: A = 1.; beta = 0.5; alpha = 0.1; gamma = 0.05; n = 1
# Control Equation is structural dynamical equation & Bouc-wen equation
# FI + FD + FS + NFS = P(t)
# Z_t = A * D_t - beta * Z * |D_t| * |Z|**(n-1) - gamma * D_t * |Z|**(n)
# z_t = A * x - beta * z * |x_t| * |z|**(n-1) - gamma * x_t * |z|**(n)
# Where:
# FI: inertia force,     M * X_tt
# FD: damping force,     C * X_t
# FS: linear resilience,  alpha * K * X
# NFS: nonlinear resilience, (1 - alpha) * K * Z
# D = {d1, d2, d3} is inter-layer displacement
# d1 = x1, d2 = x2 - x1, d3 = x3 - x2
# d1t = x1t, d2 = x2t - x1t, d3t = x3t - x2t
# output to case1.txt
import tensorflow as tf
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import gmotion, external_force

# Bouc-wen parameters
A, beta, alpha, gamma, n = 1., 0.5, 0.1, 0.05, 1
#network parameters
hidden_size = 64
lr = 1e-3
tf.set_random_seed(1)


# Structural property Matrices
def matrice(dims, stiff):
    A = np.eye(dims, dtype=np.float32)
    A[dims-1,dims-1] = stiff[-1]
    for i in range(dims-1):
        A[i,i] = stiff[i]+stiff[i+1]
        A[i,i+1] = -stiff[i+1]
    for i in range(1, dims):
        A[i,i-1] = -stiff[i-1]
    return A


dims = 3
delta_t = 1e-2
el_centro_array = np.load('./data/El_Centro_Chopra.npy')
el_centro = gmotion(el_centro_array, 0.02)
g_const = 386.0
time_range, force = external_force(el_centro, delta_t)
force *= -1*g_const

stiff, damp =np.array([100., 100., 100.]), np.array([2, 2, 2])
C = matrice(dims, damp)  # damp matrix
M = np.eye(dims, dtype=np.float32)
K = matrice(dims, stiff)*alpha  # linear stiff matrice
NK = matrice(dims, stiff)*(1-alpha)
NK[1,0] = NK[2,1] = 0.0  # nonlinear stiff matrice
NK[0,0], NK[1,1] = stiff[0]*(1-alpha), stiff[1]*(1-alpha)
EF = force[:, np.newaxis] * np.ones([1, dims])  # earthquake exited external force
EF = EF.astype(np.float32)

init = np.zeros([1, 9], dtype=np.float32)  # x1, x2, x3, y1, y2, y3, z1, z2, z3

class Actor(object):
    def __init__(self):
        self.state_init = tf.placeholder(shape=[None, 9], dtype=tf.float32)
        self.t = tf.placeholder(shape=[None,  1], dtype=tf.float32)
        # self.step = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # self.input = tf.concat([self.t, self.step], axis=1)
        self.ef = tf.placeholder(shape=[None, 3], dtype=tf.float32)
        action = self._build_mu_net('action')  # displacement
        action = (self.t + 1)*self.state_init + self.t*action

        mu_X, mu_Y, mu_Z = [action[:, 3*i:3*i+3] for i in range(3)]

        self.sigma = self._build_sigma_net('sigma')
        self.sigma = tf.clip_by_value(self.sigma, 1e-10, 0.5)

        mu_Xt1 = tf.gradients(mu_X[:,0:1], self.t)[0]  # gradient velocity
        mu_Xt2 = tf.gradients(mu_X[:,1:2], self.t)[0]  # gradient velocity
        mu_Xt3 = tf.gradients(mu_X[:,2:3], self.t)[0]  # gradient velocity
        mu_Xt = tf.concat([mu_Xt1, mu_Xt2, mu_Xt3], axis=1)

        mu_Yt1 = tf.gradients(mu_Y[:,0:1], self.t)[0]  # gradient acceleration
        mu_Yt2 = tf.gradients(mu_Y[:,0:1], self.t)[0]  # gradient acceleration
        mu_Yt3 = tf.gradients(mu_Y[:,0:1], self.t)[0]  # gradient acceleration
        mu_Yt = tf.concat([mu_Yt1, mu_Yt2, mu_Yt3], axis=1)

        mu_Zt1 = tf.gradients(mu_Z[:,0:1], self.t)[0]  # resilience derivative
        mu_Zt2 = tf.gradients(mu_Z[:,0:1], self.t)[0]  # resilience derivative
        mu_Zt3 = tf.gradients(mu_Z[:,0:1], self.t)[0]  # resilience derivative
        mu_Zt = tf.concat([mu_Zt1, mu_Zt2, mu_Zt3], axis=1)

        self.mu_Xt, self.mu_Yt, self.mu_Zt = mu_Xt, mu_Yt, mu_Zt
        self.mu = action
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        FI = tf.matmul(mu_Yt, tf.transpose(M))  # M * gradient acceleration
        FD = tf.matmul(mu_Y, tf.transpose(C))   # C * velocity
        FS = tf.matmul(mu_X, tf.transpose(K))   # K * displacement
        NFS = tf.matmul(mu_Z, tf.transpose(NK))  # NK * resilience

        d1_t = mu_Y[:,0:1]
        d2_t = mu_Y[:,1:2] - mu_Y[:,0:1]
        d3_t = mu_Y[:,2:3] - mu_Y[:,1:2]
        Dt = tf.concat([d1_t, d2_t, d3_t], axis=1)

        r1 = tf.square(FI + FD + FS + NFS - self.ef)  # equation deviation
        r2 = tf.square(mu_Xt - mu_Y)  # Newton rule deviation
        r3 = tf.square(A*Dt - beta*mu_Z*tf.abs(Dt)*tf.abs(mu_Z)**(n-1) - gamma*Dt*tf.abs(mu_Z)**n - mu_Zt)  # Bouc-wen deviation
        self.R = tf.reduce_sum([r1, r2, r3])  # total loss

        self.log_prob = self.normal_dist.log_prob(self.mu)
        self.entropy = self.normal_dist.entropy()
        self.loss = tf.reduce_sum(1e2*self.entropy + self.log_prob*self.R)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def _build_mu_net(self, scope):
        with tf.name_scope(scope):
            ly1 = tf.layers.dense(inputs=self.t, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly2 = tf.layers.dense(inputs=ly1, units=hidden_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly3 = tf.layers.dense(inputs=ly2, units=hidden_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
        return tf.layers.dense(inputs=ly3, units=9, activation=None)  # returns mu value of x

    def _build_sigma_net(self, scope):
        with tf.name_scope(scope):
            ly1 = tf.layers.dense(inputs=self.t, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
        return tf.layers.dense(inputs=ly1, units=9, activation=None)  # returns sigma value of [x, y, xt, yt]

tf.get_default_graph()
sess = tf.Session()
actor = Actor()
sess.run(tf.global_variables_initializer())


solutions, all_steps, solvingtime, physical_derivatives = [], [], [], []
solution = init
solutions.append(solution)
all_steps.append(0)
solvingtime.append(0.)
start0 = start = time.time()
step = 0

error_thr = 1e-3

for step in range(len(EF)):
    error = 1.
    ep_steps = 0
    while error > error_thr:
        t = np.ones([1,1])*delta_t
        ef = EF[step][np.newaxis,:]

        mu, sigma, R, loss, _, xt, yt, zt = sess.run([actor.mu, actor.sigma, actor.R, actor.loss, actor.train_op,
                                                      actor.mu_Xt, actor.mu_Yt, actor.mu_Zt], feed_dict={
            actor.t:t, actor.ef:ef, actor.state_init:solution})
        error = R

        ep_steps += 1
        if ep_steps % 1000 == 0:
            print('*********************************************')
            print('step-%i, ep_step-%i, :: time: %0.2f, error:%0.2e, aloss:%0.2f'
                  %(step, ep_steps, time.time()-start, error, loss))
            print(mu[0],'\n',sigma[0])
        # if ep_steps > 10000:
        #     sess.run(tf.global_variables_initializer())
        #     ep_steps = 1
    solution = mu
    derivatives = np.concatenate([xt, yt, zt], axis=1)

    solutions.append(solution[0].tolist())
    physical_derivatives.append(derivatives[0].tolist())
    all_steps.append(ep_steps)
    solvingtime.append(time.time() - start)
    print('step-%i, ep_step-%i, :: time: %0.2f, error:%0.2e, aloss:%0.2f'
          % (step, ep_steps, solvingtime[-1], error, loss))
    print(mu[-1])
    start = time.time()

np.save('./result/solution_MDOF.npy', solutions)
np.save('./result/performance_MDOF.npy', (solvingtime, all_steps))
sio.savemat('./result/solution_MDOF.mat', {'data':solutions})
sio.savemat('./result/performance_MDOF.mat', {'data':solvingtime})

