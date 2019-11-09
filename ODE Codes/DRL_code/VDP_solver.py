# -*- coding: utf-8 -*--
# case 1-1
# parameters: alpha=0.1; beta = 0;  solution domain = [0, 100]
# delta_t = 1e-2;  initial condition = [1., 1.];
# output to case1.txt
import tensorflow as tf
import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
import scipy.io as sio

# system setting: parameters & initial condition
alpha, beta, omega = 5., 5., 7.
init = [0., 0.]
P = 1.
error_thr = 1e-3
# network setting
lr = 1e-3
delta_t = 1e-4
hidden_size = 32
entropy_par = 1.
tf.set_random_seed(1)
n_steps = 100

class Actor(object):  # continuous PG_solver for VDP problem
    def __init__(self):
        self.state_init = tf.placeholder(shape=[3], dtype=tf.float32)   # feed_in initial solution
        x0, y0, yt0 = [self.state_init[i] for i in range(3)]  # initial phase
        self.t = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # feed_in time t
        self.step = tf.placeholder(shape=[None, 1], dtype=tf.float32)  # feed_in time step
        self.input = tf.concat([self.t, self.step], axis=1)
        # self.entropy_par = tf.placeholder(shape=[1], dtype=tf.float32)

        action = self._build_mu_net('action')
        X, Y = [action[:, i:i+1] for i in range(2)]
        # mu_x = tf.exp(-self.t)*x0 + self.t*X
        mu_x = (self.t+1)*x0 + self.t*X
        # mu_y = tf.exp(-self.t)*y0 + self.t*Y
        mu_y = (self.t+1)*y0 + self.t*Y
        mu_xt = tf.gradients(mu_x, self.t)[0]
        mu_yt = tf.gradients(mu_y, self.t)[0]
        self.mu = tf.concat([mu_x, mu_y, mu_yt], axis=1)
        self.sigma = self._build_sigma_net('sigma')
        self.sigma = tf.clip_by_value(self.sigma, 1e-20, 1.)

        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        self.R = tf.square(mu_yt - alpha*(1-mu_x**2)*mu_y + mu_x - beta*P*tf.cos(omega*self.step*delta_t+self.t)) + tf.square(mu_xt - mu_y)
        self.log_prob = self.normal_dist.log_prob(self.mu)
        entropy = self.normal_dist.entropy()
        self.loss = tf.reduce_mean(1e-1*entropy + self.log_prob*self.R)
        # self.loss = tf.reduce_mean(self.log_prob*self.R)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def _build_mu_net(self, scope):
        with tf.name_scope(scope):
            ly1 = tf.layers.dense(inputs=self.t, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly2 = tf.layers.dense(inputs=ly1, units=hidden_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly3 = tf.layers.dense(inputs=ly2, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
        return tf.layers.dense(inputs=ly3, units=2, activation=None)  # returns mu value of x

    def _build_sigma_net(self, scope):
        with tf.name_scope(scope):
            ly1 = tf.layers.dense(inputs=self.t, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
        return tf.layers.dense(inputs=ly1, units=3, activation=None)  # returns sigma value of [x, y, xt, yt]


tf.get_default_graph()
sess = tf.Session()
actor = Actor()
sess.run(tf.global_variables_initializer())

solutions, all_steps, solvingtime = [], [], []
x_i, y_i = init
solution = [x_i, y_i, alpha*(1-x_i**2)*y_i - x_i + beta*P]
solutions.append(solution)
all_steps.append(0)
solvingtime.append(0.)
start0 = start = time.time()
step = 0


for step in range(int(50/delta_t)):
    error = 1.
    ep_steps = 0
    # entropy_par = 1e2
    while error > error_thr:
        # t = np.ones([1,1])*delta_t
        t = np.linspace(0., delta_t, n_steps)
        t = t[:, np.newaxis]

        mu, sigma, R, loss, _ = sess.run([actor.mu, actor.sigma, actor.R, actor.loss, actor.train_op], feed_dict={
            actor.state_init:solution, actor.t:t, actor.step:np.ones([n_steps,1])*step})
        error = np.mean(R)

        ep_steps += 1
        if ep_steps % 1000 == 0:
            print('*********************************************')
            print('step-%i, ep_step-%i, :: time: %0.2f, error:%0.2e, aloss:%0.2f'
                  %(step, ep_steps, time.time()-start, error, loss))
            print(mu[0],'\n',sigma[0])

    solution = mu[-1]
    solutions.append(solution.tolist())
    all_steps.append(ep_steps)
    solvingtime.append(time.time()-start)
    print('step-%i, ep_step-%i, :: time: %0.2f, error:%0.2e, aloss:%0.2f'
          % (step, ep_steps, solvingtime[-1], error, loss))
    # print(mu[-1])
    start = time.time()

print(time.time()-start0)
plt.plot(all_steps)
data = np.vstack(solutions)
plt.figure()
plt.plot(data[:,0], data[:,1])

data = np.vstack(solutions)

np.save('./result/solution_case3-1.npy', data)
np.save('./result/performance_case3-1.npy', (all_steps, solvingtime))
sio.savemat('./result/solution_case3-1.mat', {'data':solutions})
sio.savemat('./result/performance_case3-1.mat', {'time':solvingtime, 'iters':all_steps})
