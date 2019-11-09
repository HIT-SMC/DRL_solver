# -*- coding: utf-8 -*--
# U = (t+1)U0 + tF(x)
import numpy as np
import tensorflow as tf
from Lorenz_continuous_environment import Lorenz
import matplotlib.pyplot as plt
import time, os
import scipy.io as sio
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.set_random_seed(0)
np.random.seed(0)

params = [10., 8./3, 10.]
init = [0., -2., 0.]
hidden_size = 64
lr = 1e-3
delta_t = 1e-3  # time interval


class Actor(object):
    def __init__(self, sess):
        self.sess = sess
        env = Lorenz(pars=params)
        # self.state = env._reset(init=init)
        self.state_init = tf.placeholder(shape=[6], dtype=tf.float32)
        alpha, beta, rho = env.sigma, env.beta, env.r
        # x0, y0, z0, xt0, yt0, zt0, _ = self.state  # initial condition
        x0, y0, z0, xt0, yt0, zt0 = [self.state_init[i] for i in range(6)]  # initial condition

        self.t = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # self.var = tf.placeholder(shape=[None, 6], dtype=tf.float32)
        self.entropy_par = tf.placeholder(shape=None, dtype=tf.float32)
        self.learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

        # self.t = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        mu_x = self._build_mu_net('mu_x')
        mu_y = self._build_mu_net('mu_x')
        mu_z = self._build_mu_net('mu_x')
        X = (self.t + 1)*x0 + self.t*mu_x
        Y = (self.t + 1)*y0 + self.t*mu_y
        Z = (self.t + 1)*z0 + self.t*mu_z

        mu_xt = tf.gradients(X, self.t)[0]
        mu_yt = tf.gradients(Y, self.t)[0]
        mu_zt = tf.gradients(Z, self.t)[0]
        self.mu = tf.concat([X, Y, Z, mu_xt, mu_yt, mu_zt], axis=1)
        self.sigma = self._build_sigma_net('sigma')
        self.sigma = tf.clip_by_value(self.sigma, 1e-20, 1e2)

        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        # self.normal_dist = tf.distributions.Normal(self.mu, self.var)

        self.R = tf.square(mu_xt - alpha*(Y - X)) + \
                 tf.square(mu_yt - rho*X + Y + X*Z) + \
                 tf.square(mu_zt + beta*Z - X*Y)

        self.log_prob = self.normal_dist.log_prob(self.mu)
        exp_v = self.normal_dist.entropy()
        # self.loss = tf.reduce_mean(1e3*exp_v + self.log_prob*self.R/1e3) # 使用固定的entropy 参数
        self.loss = tf.reduce_mean(self.entropy_par*exp_v + self.log_prob*self.R/1e3)  # 使用feed_in的entropy参数

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)  # 使用feed_in的learning rate
        # self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)  #使用固定的learning rate
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
            ly4 = tf.layers.dense(inputs=ly3, units=hidden_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly5 = tf.layers.dense(inputs=ly4, units=hidden_size, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            ly6 = tf.layers.dense(inputs=ly5, units=hidden_size//2, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0., .1),
                                  bias_initializer=tf.constant_initializer(0.))
            a_mu = tf.layers.dense(inputs=ly6, units=1, activation=None)
        return a_mu
    def _build_sigma_net(self, scope):
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
            a_sigma = tf.layers.dense(inputs=ly3, units=6, activation=tf.nn.softplus)
        return a_sigma


env = Lorenz()
tf.get_default_graph()
sess = tf.Session()
actor = Actor(sess)
sess.run(tf.global_variables_initializer())

start = time.time()
t0 = np.zeros([1,1])
saver = tf.train.Saver()

trajectory, reward_history, solvingtime_history = [], [], []
S = env._reset(init)
state = S[:-1]
trajectory.append(state)
reward_history.append(0.)
solvingtime_history.append(0.)

for step in range(int(100/delta_t)):
    error = 1.
    entropy_par, lr = np.array(1e2, dtype=np.float32), np.array(1e-2, dtype=np.float32)  # initial params
    ep_steps = 0
    while error > 1e-6:
        # t = np.ones([1,1])*(step+1)*delta_t
        t = np.ones([1,1])*delta_t
        mu, sigma, R, loss, log_prob, _ = sess.run([actor.mu, actor.sigma, actor.R, actor.loss, actor.log_prob, actor.train_op], feed_dict={
            actor.state_init:state, actor.t: t, actor.entropy_par: entropy_par, actor.learning_rate: lr})
        error = R[0,0]
        if ep_steps % 1000 == 0:
            print('*******************')
            # mu, sigma = sess.run([actor.mu, actor.sigma], feed_dict={actor.t: t0})
            # print('ep_step:%i, time:%0.2f, error:%0.2e, aloss:%0.2f, lr:%2.2e, entro:%2.2e' %(ep_step, time.time()-start, error, aloss, lr, entropy_par))
            print('step: %i, ep_step:%i, time:%0.2f, error:%0.2e, aloss:%0.2f' % (step, ep_steps, time.time() - start, error, loss))
            print(mu, '\n')
            print(sigma)
            print('\n', lr, entropy_par)
            if sigma[0, 0] < 1e-5: #  and entropy_par > 1e-1:
                entropy_par = 10.
            if sigma[0,0] < 1e-5 and lr>1e-4:
                lr *= 0.999
        ep_steps += 1

        if time.time() - start > 10:
            entropy_par, lr = np.array(1e2, dtype=np.float32), np.array(1e-2, dtype=np.float32)  # initial params
            actor.sess.run(tf.global_variables_initializer())
            start = time.time()
    trajectory.append(mu[0])
    reward_history.append(-error)
    solvingtime_history.append(time.time()-start)
    start = time.time()
    state = mu[0]

# saver.save(sess, './discrete_continuous_solution-1.cpkt')
np.save('./result-11.npy', {'reward_history':reward_history, 'trajectory': trajectory, 'solvingtime_history':solvingtime_history})
sio.savemat('./result-11.mat', {'reward_history':reward_history, 'trajectory': trajectory, 'solvingtime_history':solvingtime_history})
