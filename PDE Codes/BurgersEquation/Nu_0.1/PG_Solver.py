"""
DRL Solver for Burgers' Equation
ut+u*ux-Nu*uxx = 0
Author: Shiyin Wei, Xiaowei Jin, Hui Li*
Date: April, 2018
"""

import tensorflow as tf
import numpy as np
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# ********** GPU ratio ********** #
GR = 0.2

# ********** PDE parameters ********** #
Nu = 0.1

# ********** Hyper parameters ********** #
LR_A_ini = 5e-4  # Initial learning rate for actor
tor = 5e-5
forward_T_n = 9

# ********** Points sampling ********** #
Num = 99
S_x_ = np.linspace(-1., 1., Num + 2)
S_x = S_x_[1:-1][:, np.newaxis]
deltaT = 0.01


# ********** DP ********** #
class DP(object):
    def __init__(self, a_dim, s_dim):
        self.pointer = 0
        self.lr = tf.placeholder(tf.float32, None, 'LR')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GR
        self.sess = tf.Session(config=config)
        self.on_train = tf.constant(True, dtype=bool)
        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')

        self.R_e = tf.placeholder(tf.float32, [None, 1], 'r_e')

        self.a = self._build_a(self.S)
        self.normal_dist = tf.distributions.Normal(self.a[:, 0], self.a[:, 4])
        self.normal_dist_prob = tf.expand_dims(self.normal_dist.prob(self.a[:, 0]), 1)
        self.pi_e = self.normal_dist_prob

        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')

        R_e = self.a[:, 1:2] + self.a[:, 0:1] * self.a[:, 2:3] - Nu * self.a[:, 3:4]

        self.q_e = tf.reduce_mean(tf.square(R_e))

        self.a_loss = tf.reduce_mean(tf.square(R_e)*self.pi_e)
        self.atrain = tf.train.AdamOptimizer(self.lr).minimize(self.a_loss, var_list=self.a_params)

        self.sess.run(tf.global_variables_initializer())

    def learn_a(self, learn_rate, bs):
        self.sess.run(self.atrain, {self.S: bs, self.lr: learn_rate})
        aloss = self.sess.run(self.a_loss, {self.S: bs})
        aloss_e = self.sess.run(self.q_e, {self.S: bs})
        return aloss, aloss_e

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            w1 = tf.get_variable('w1', [2, 32],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b1 = tf.get_variable('b1', 32, initializer=tf.constant_initializer(0.), trainable=trainable)
            l1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(s, w1), b1))

            w2 = tf.get_variable('w2', [32, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b2 = tf.get_variable('b2', 64, initializer=tf.constant_initializer(0.), trainable=trainable)
            l2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l1, w2), b2))

            w3 = tf.get_variable('w3', [64, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b3 = tf.get_variable('b3', 64, initializer=tf.constant_initializer(0.), trainable=trainable)
            l3 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l2, w3), b3))

            w4 = tf.get_variable('w4', [64, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b4 = tf.get_variable('b4', 64, initializer=tf.constant_initializer(0.), trainable=trainable)
            l4 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l3, w4), b4))

            w5 = tf.get_variable('w5', [64, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b5 = tf.get_variable('b5', 64, initializer=tf.constant_initializer(0.), trainable=trainable)
            l5 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l4, w5), b5))

            w6 = tf.get_variable('w6', [64, 64],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b6 = tf.get_variable('b6', 64, initializer=tf.constant_initializer(0.), trainable=trainable)
            l6 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l5, w6), b6))

            w7 = tf.get_variable('w7', [64, 32],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b7 = tf.get_variable('b7', 32, initializer=tf.constant_initializer(0.), trainable=trainable)
            l7 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l6, w7), b7))

            u_ = tf.layers.dense(l7, 1, name='a_action', trainable=trainable)
            sigma = tf.nn.sigmoid(tf.layers.dense(l7, 1, name='a_sigma', trainable=trainable))

            u = (s[:, 1:] + 1.)*(s[:, 1:] - 1.)*s[:, 0:1]*u_ - tf.sin(np.pi*s[:, 1:])

            u_t = tf.expand_dims(tf.gradients(u[:], s)[0][:, 0], 1)
            u_x = tf.expand_dims(tf.gradients(u[:], s)[0][:, 1], 1)
            u_xx = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(u[:], s)[0][:, 1], 1), s)[0][:, 1], 1)

            action = tf.concat([u, u_t, u_x, u_xx, sigma], axis=1)

            return action


# ********** TRAINING ********** #
s_dim_ = 2
a_dim_ = 4

dp = DP(a_dim_, s_dim_)

t1 = time.time()

sess = tf.Session()
saver = tf.train.Saver()

start = time.time()

time_0 = time.time()
init_now = -np.sin(np.pi*S_x)
err = []
IteNo = []
Col = init_now

for ti in range(100):
    forward_T_n += 1
    tn = (ti + 1)*deltaT

    S_t_now_1 = np.reshape(np.random.rand((forward_T_n-1)*Num), [(forward_T_n-1)*Num, 1])*tn
    S_t_now_2 = tn*np.ones([Num, 1], dtype=np.float32)
    S_x_1 = np.reshape(np.random.rand((forward_T_n-1)*Num), [(forward_T_n-1)*Num, 1])*2. - 1.
    S_t_now_1_all = np.concatenate([S_t_now_1, S_x_1], axis=1)
    S_t_now_2_all = np.concatenate([S_t_now_2, S_x], axis=1)
    # S_t_now = deltaT*np.ones([Num, 1], dtype=np.float32)

    S_all_now = np.concatenate([S_t_now_1_all, S_t_now_2_all], axis=0)
    S_all = S_all_now
    if ti > 1:
        LR_A = 0.5*LR_A_ini
    else:
        LR_A = LR_A_ini

    error = 1e4
    ep_step = 0
    while error > tor:
        time_1 = time.time()
        ep_step += 1
        if ep_step % 100 == 0:
            LR_A *= 0.98
            # Lamb_I *= 0.9
            if LR_A < 1e-5:
                LR_A = 1e-5
            # if Lamb_I < 1.:
            #     Lamb_I = 1.
        a_loss,  aloss_e = dp.learn_a(LR_A, S_all)

        error = aloss_e
        print('')
        print('[Time_step] %d [Iteration_step] %d ************************** '
              '[Iteration_time] %10.8f' % (ti + 1, ep_step, time.time() - time_1))
        print('[error] %10.8f  [a_loss] %10.8f' % (error, a_loss))
        print('[Learning rate] %10.8f' % LR_A)
        print('')
        err.append([a_loss, aloss_e])
    init_now = dp.sess.run(dp.a[-Num:, 0:1], {dp.S: S_all})
    Col = np.concatenate([Col, init_now], axis=1)
    IteNo.append(ep_step)

    np.savetxt('err.dat', err)
    np.savetxt('Col.dat', Col)
    np.savetxt('IteNo.dat', IteNo)

    print('Running time: ', time.time() - t1)

    saver.save(dp.sess, './model/dpnet_result.cpkt')
