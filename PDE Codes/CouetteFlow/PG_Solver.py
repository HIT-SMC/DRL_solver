"""
DRL Solver for Navier-Stokes Equation
ut + u*div(u) = -1/rho*laplace(p) + niu*laplace(u)
Author: Shiyin Wei, Xiaowei Jin, Hui Li*
Date: April, 2018
"""

import tensorflow as tf
import numpy as np
import time
import scipy.io as sio

# ********** GPU ratio ********** #
GR = 0.5

# ********** hyper parameters ********** #
LR_A = 0.001  # learning rate for actor
Lamb_B = 500.
Lamb_B_ = 500.
Lamb_I = 1.
Lamb_I_ = 1.
ep_step_initial = 0
step_all = 1e5
tor = 5e-5

boundary = sio.loadmat('./data/S_B.mat')
S_B = boundary['S_B']

boundary_ = sio.loadmat('./data/S_B_.mat')
S_B_ = boundary_['S_B_']

initial = sio.loadmat('./data/S_initial.mat')
S_initial = initial['S_initial']

initial_ = sio.loadmat('./data/S_initial_.mat')
S_initial_ = initial_['S_initial_']

inter = sio.loadmat('./data/S_inter.mat')
S_inter = inter['S_inter']

Sampling_Num_b = S_B.shape[0]
Sampling_Num_i = S_initial.shape[0]
Sampling_Num_e = S_inter.shape[0]
BATCH_N_b = 1
BATCH_N_i = 1
BATCH_N_e = 1
BATCH_SIZE_b = Sampling_Num_b * BATCH_N_b
BATCH_SIZE_i = Sampling_Num_i * BATCH_N_i
BATCH_SIZE_e = Sampling_Num_e * BATCH_N_e


# ********** PG ********** #
class PG(object):
    def __init__(self, a_dim, s_dim):
        self.lr = tf.placeholder(tf.float32, None, 'LR')
        self.Lamb_B = tf.placeholder(tf.float32, None, 'L_B')
        self.Lamb_B_ = tf.placeholder(tf.float32, None, 'L_B_')
        self.Lamb_I = tf.placeholder(tf.float32, None, 'L_I')
        self.Lamb_I_ = tf.placeholder(tf.float32, None, 'L_I')
        self.rho = 0.002377  # slug/ft^3
        self.miu = 1.7994e-5 / 14.593903 / 3.2808399  # slug/(ft*s)
        self.ue = 1.
        self.pointer = 0
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GR
        self.sess = tf.Session(config=config)
        self.on_train = tf.constant(True, dtype=bool)
        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_b = self.S[:BATCH_SIZE_b, :]
        self.S_b_ = self.S[BATCH_SIZE_b:BATCH_SIZE_b*2, :]
        self.S_i = self.S[BATCH_SIZE_b*2:(BATCH_SIZE_b*2 + BATCH_SIZE_i), :]
        self.S_i_ = self.S[(BATCH_SIZE_b*2 + BATCH_SIZE_i):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2), :]

        self.R_b = tf.placeholder(tf.float32, [None, 1], 'r_b')
        self.R_i = tf.placeholder(tf.float32, [None, 1], 'r_i')
        self.R_e = tf.placeholder(tf.float32, [None, 1], 'r_e')

        self.a = self._build_a(self.S)
        self.normal_dist_u = tf.distributions.Normal(self.a[:, 0], self.a[:, 13])
        self.normal_dist_prob_u = tf.expand_dims(self.normal_dist_u.prob(self.a[:, 0]), 1)
        self.normal_dist_v = tf.distributions.Normal(self.a[:, 1], self.a[:, 14])
        self.normal_dist_prob_v = tf.expand_dims(self.normal_dist_u.prob(self.a[:, 1]), 1)
        self.normal_dist_p = tf.distributions.Normal(self.a[:, 2], self.a[:, 15])
        self.normal_dist_prob_p = tf.expand_dims(self.normal_dist_p.prob(self.a[:, 2]), 1)

        self.a_b = self.a[:BATCH_SIZE_b, :]
        self.pi_b_u = self.normal_dist_prob_u[:BATCH_SIZE_b, :]
        self.pi_b_v = self.normal_dist_prob_v[:BATCH_SIZE_b, :]
        self.pi_b_p = self.normal_dist_prob_p[:BATCH_SIZE_b, :]
        self.a_b_ = self.a[BATCH_SIZE_b:BATCH_SIZE_b*2, :]
        self.pi_b_u_ = self.normal_dist_prob_u[BATCH_SIZE_b:BATCH_SIZE_b*2, :]
        self.pi_b_v_ = self.normal_dist_prob_v[BATCH_SIZE_b:BATCH_SIZE_b*2, :]
        self.pi_b_p_ = self.normal_dist_prob_p[BATCH_SIZE_b:BATCH_SIZE_b*2, :]
        self.a_i = self.a[BATCH_SIZE_b*2:(BATCH_SIZE_b*2 + BATCH_SIZE_i), :]
        self.pi_i_u = self.normal_dist_prob_u[BATCH_SIZE_b*2:(BATCH_SIZE_b*2 + BATCH_SIZE_i), :]
        self.pi_i_v = self.normal_dist_prob_v[BATCH_SIZE_b*2:(BATCH_SIZE_b*2 + BATCH_SIZE_i), :]
        self.pi_i_p = self.normal_dist_prob_p[BATCH_SIZE_b*2:(BATCH_SIZE_b*2 + BATCH_SIZE_i), :]
        self.a_i_ = self.a[(BATCH_SIZE_b*2 + BATCH_SIZE_i):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2), :]
        self.pi_i_u_ = self.normal_dist_prob_u[(BATCH_SIZE_b*2 + BATCH_SIZE_i):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2), :]
        self.pi_i_v_ = self.normal_dist_prob_v[(BATCH_SIZE_b*2 + BATCH_SIZE_i):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2), :]
        self.pi_i_p_ = self.normal_dist_prob_p[(BATCH_SIZE_b*2 + BATCH_SIZE_i):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2), :]
        self.a_e = self.a[(BATCH_SIZE_b*2 + BATCH_SIZE_i*2):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2 + BATCH_SIZE_e), :]
        self.pi_e_u = self.normal_dist_prob_u[(BATCH_SIZE_b*2 + BATCH_SIZE_i*2):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2 + BATCH_SIZE_e), :]
        self.pi_e_v = self.normal_dist_prob_v[(BATCH_SIZE_b*2 + BATCH_SIZE_i*2):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2 + BATCH_SIZE_e), :]
        self.pi_e_p = self.normal_dist_prob_p[(BATCH_SIZE_b*2 + BATCH_SIZE_i*2):(BATCH_SIZE_b*2 + BATCH_SIZE_i*2 + BATCH_SIZE_e), :]
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')

        self.pi_b = tf.reduce_mean(self.pi_b_u + self.pi_b_v + self.pi_b_p)

        self.pi_b_ = tf.reduce_mean(self.pi_b_u_ + self.pi_b_v_ + self.pi_b_p_)

        self.pi_i = tf.reduce_mean(self.pi_i_u + self.pi_i_v + self.pi_i_p)

        self.pi_i_ = tf.reduce_mean(self.pi_i_u_ + self.pi_i_v_ + self.pi_i_p_)

        self.pi_e = tf.reduce_mean(self.pi_e_u + self.pi_e_v + self.pi_e_p)

        R_b = tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 0], 1))) + \
              tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 1], 1))) + tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 2], 1))) + \
              0. *tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 11], 1))) + 0.*tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 12], 1))) + \
              tf.reduce_mean(tf.square(tf.expand_dims(self.a_b[:, 3], 1) + tf.expand_dims(self.a_b[:, 9], 1)))
        R_b_ = tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 0], 1) - self.ue)) + \
               tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 1], 1))) + tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 2], 1))) \
               + 0.*tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 11], 1))) + 0.*tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 12], 1))) + \
               tf.reduce_mean(tf.square(tf.expand_dims(self.a_b_[:, 3], 1) + tf.expand_dims(self.a_b_[:, 9], 1)))

        R_i = tf.reduce_mean(tf.square(tf.expand_dims(self.a_i[:, 1], 1))) + tf.reduce_mean(tf.square(tf.expand_dims(self.a_i[:, 2], 1))) + \
              tf.reduce_mean(tf.square(tf.expand_dims(self.a_i[:, 3], 1) + tf.expand_dims(self.a_i[:, 9], 1)))
        R_i_ = tf.reduce_mean(tf.square(tf.expand_dims(self.a_i_[:, 2], 1))) + \
               tf.reduce_mean(tf.square(tf.expand_dims(self.a_i_[:, 3], 1) + tf.expand_dims(self.a_i_[:, 9], 1)))

        R_e_u = tf.expand_dims(self.a_e[:, 0], 1)*tf.expand_dims(self.a_e[:, 3], 1) + tf.expand_dims(self.a_e[:, 1], 1)*tf.expand_dims(self.a_e[:, 5], 1) + \
                1./self.rho*tf.expand_dims(self.a_e[:, 11], 1) - self.miu/self.rho*(tf.expand_dims(self.a_e[:, 4], 1) + tf.expand_dims(self.a_e[:, 6], 1))
        R_e_v = tf.expand_dims(self.a_e[:, 0], 1)*tf.expand_dims(self.a_e[:, 7], 1) + tf.expand_dims(self.a_e[:, 1], 1)*tf.expand_dims(self.a_e[:, 9], 1) + \
                1./self.rho*tf.expand_dims(self.a_e[:, 12], 1) - self.miu/self.rho*(tf.expand_dims(self.a_e[:, 8], 1) + tf.expand_dims(self.a_e[:, 10], 1))

        R_e = tf.reduce_mean(tf.square(R_e_u) + tf.square(R_e_v)) + \
              0.*tf.reduce_mean(tf.square(tf.expand_dims(self.a_e[:, 11], 1))) + 0.*tf.reduce_mean(tf.square(tf.expand_dims(self.a_e[:, 12], 1))) + \
              tf.reduce_mean(tf.square(tf.expand_dims(self.a_e[:, 3], 1) + tf.expand_dims(self.a_e[:, 9], 1)))

        self.q_b = R_b*self.Lamb_B*self.pi_b*BATCH_N_b
        self.q_b_ = R_b_*self.Lamb_B_*self.pi_b_*BATCH_N_b
        self.q_i = R_i*self.Lamb_I*self.pi_i*BATCH_N_i
        self.q_i_ = R_i_*self.Lamb_I_*self.pi_i_*BATCH_N_i
        self.q_e = R_e*self.pi_e*BATCH_SIZE_e
        self.a_loss = (self.q_b + self.q_b_ + self.q_i + self.q_i_ + self.q_e)/(BATCH_N_b*2. + BATCH_N_i*2. + BATCH_N_e)

        self.atrain = tf.train.AdamOptimizer(self.lr).minimize(self.a_loss, var_list=self.a_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a_b, {self.S: s}), \
               self.sess.run(self.a_b_, {self.S: s}), \
               self.sess.run(self.a_i, {self.S: s}), \
               self.sess.run(self.a_i_, {self.S: s}), \
               self.sess.run(self.a_e, {self.S: s})

    def learn_a(self, learn_rate, lamb_b, lamb_b_, lamb_i, lamb_i_, bs):

        self.sess.run(self.atrain, {self.S: bs, self.lr: learn_rate, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss = self.sess.run(self.a_loss, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss_b = self.sess.run(self.q_b, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss_b_ = self.sess.run(self.q_b_, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss_i = self.sess.run(self.q_i, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss_i_ = self.sess.run(self.q_i_, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        aloss_e = self.sess.run(self.q_e, {self.S: bs, self.Lamb_B: lamb_b, self.Lamb_B_: lamb_b_, self.Lamb_I: lamb_i, self.Lamb_I_: lamb_i_})
        return aloss, aloss_b, aloss_b_, aloss_i, aloss_i_, aloss_e

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            w1 = tf.get_variable('w1', [2, 20],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b1 = tf.get_variable('b1', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(s, w1), b1))

            w2 = tf.get_variable('w2', [20, 30],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b2 = tf.get_variable('b2', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l1, w2), b2))

            w3 = tf.get_variable('w3', [30, 40],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b3 = tf.get_variable('b3', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l3 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l2, w3), b3))

            w4 = tf.get_variable('w4', [40, 40],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b4 = tf.get_variable('b4', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l4 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l3, w4), b4))

            w5 = tf.get_variable('w5', [40, 40],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b5 = tf.get_variable('b5', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l5 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l4, w5), b5))

            w6 = tf.get_variable('w6', [40, 30],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b6 = tf.get_variable('b6', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l6 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l5, w6), b6))

            w7 = tf.get_variable('w7', [30, 20],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b7 = tf.get_variable('b7', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l7 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l6, w7), b7))

            a_u = tf.layers.dense(l7, 1, name='a_1u', trainable=trainable)
            sigma_u = tf.nn.sigmoid(tf.layers.dense(l7, 1, name='sigma_u', trainable=trainable))

            w1v = tf.get_variable('w1v', [2, 20],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b1v = tf.get_variable('b1v', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l1v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(s, w1v), b1v))

            w2v = tf.get_variable('w2v', [20, 30],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b2v = tf.get_variable('b2v', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l2v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l1v, w2v), b2v))

            w3v = tf.get_variable('w3v', [30, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b3v = tf.get_variable('b3v', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l3v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l2v, w3v), b3v))

            w4v = tf.get_variable('w4v', [40, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b4v = tf.get_variable('b4v', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l4v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l3v, w4v), b4v))

            w5v = tf.get_variable('w5v', [40, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b5v = tf.get_variable('b5v', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l5v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l4v, w5v), b5v))

            w6v = tf.get_variable('w6v', [40, 30],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b6v = tf.get_variable('b6v', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l6v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l5v, w6v), b6v))

            w7v = tf.get_variable('w7v', [30, 20],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b7v = tf.get_variable('b7v', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l7v = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l6v, w7v), b7v))

            a_v = tf.layers.dense(l7v, 1, name='a_2v', trainable=trainable)
            sigma_v = tf.nn.sigmoid(tf.layers.dense(l7v, 1, name='sigma_v', trainable=trainable))

            w1p = tf.get_variable('w1p', [2, 20],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b1p = tf.get_variable('b1p', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l1p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(s, w1p), b1p))

            w2p = tf.get_variable('w2p', [20, 30],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b2p = tf.get_variable('b2p', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l2p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l1p, w2p), b2p))

            w3p = tf.get_variable('w3p', [30, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b3p = tf.get_variable('b3p', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l3p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l2p, w3p), b3p))

            w4p = tf.get_variable('w4p', [40, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b4p = tf.get_variable('b4p', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l4p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l3p, w4p), b4p))

            w5p = tf.get_variable('w5p', [40, 40],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b5p = tf.get_variable('b5p', 40, initializer=tf.constant_initializer(0.), trainable=trainable)
            l5p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l4p, w5p), b5p))

            w6p = tf.get_variable('w6p', [40, 30],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b6p = tf.get_variable('b6p', 30, initializer=tf.constant_initializer(0.), trainable=trainable)
            l6p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l5p, w6p), b6p))

            w7p = tf.get_variable('w7p', [30, 20],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=trainable)
            b7p = tf.get_variable('b7p', 20, initializer=tf.constant_initializer(0.), trainable=trainable)
            l7p = tf.nn.tanh(tf.nn.bias_add(tf.matmul(l6p, w7p), b7p))

            a_p = tf.layers.dense(l7p, 1, name='a_3p', trainable=trainable)
            sigma_p = tf.nn.sigmoid(tf.layers.dense(l7p, 1, name='sigma_p', trainable=trainable))

            u_x = tf.expand_dims(tf.gradients(a_u[:, 0], s)[0][:, 0], 1)
            u_xx = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(a_u[:, 0], s)[0][:, 0], 1),
                                               s)[0][:, 0], 1)
            u_y = tf.expand_dims(tf.gradients(a_u[:, 0], s)[0][:, 1], 1)
            u_yy = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(a_u[:, 0], s)[0][:, 1], 1),
                                               s)[0][:, 1], 1)

            v_x = tf.expand_dims(tf.gradients(a_v[:, 0], s)[0][:, 0], 1)
            v_xx = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(a_v[:, 0], s)[0][:, 0], 1),
                                               s)[0][:, 0], 1)
            v_y = tf.expand_dims(tf.gradients(a_v[:, 0], s)[0][:, 1], 1)
            v_yy = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(a_v[:, 0], s)[0][:, 1], 1),
                                               s)[0][:, 1], 1)

            p_x = tf.expand_dims(tf.gradients(a_p[:, 0], s)[0][:, 0], 1)

            p_y = tf.expand_dims(tf.gradients(a_p[:, 0], s)[0][:, 1], 1)

            action = tf.concat([a_u, a_v, a_p, u_x, u_xx, u_y, u_yy, v_x, v_xx,
                                v_y, v_yy, p_x, p_y, sigma_u, sigma_v, sigma_p], axis=1)
            return action


# ********** TRAINING ********** #
s_dim_ = 2
a_dim_ = 16

pg = PG(a_dim_, s_dim_)

rho = 0.002377  # slug/ft^3
miu = 1.7994e-5 / 14.593903 / 3.2808399  # slug/(ft*s)
t1 = time.time()
err = []
err_b = []
err_b_ = []
err_i = []
err_i_ = []
err_e = []
Ua = np.zeros([26000, 1])
S_B_all = S_B
S_B_all_ = S_B_
S_initial_all = S_initial
S_initial_all_ = S_initial_
S_inter_all = S_inter
for j1 in range(BATCH_N_b - 1):
    S_B_all = np.concatenate([S_B_all, S_B], axis=0)
    S_B_all_ = np.concatenate([S_B_all_, S_B_], axis=0)
for j2 in range(BATCH_N_i - 1):
    S_initial_all = np.concatenate([S_initial_all, S_initial], axis=0)
    S_initial_all_ = np.concatenate([S_initial_all_, S_initial_], axis=0)
for j3 in range(BATCH_N_e - 1):
    S_inter_all = np.concatenate([S_inter_all, S_inter], axis=0)
S_all = np.concatenate([S_B_all, S_B_all_, S_initial_all, S_initial_all_, S_inter_all], axis=0)
sess = tf.Session()
saver = tf.train.Saver()

start = time.time()
error = 1e4
ep_step = ep_step_initial
time_0 = time.time()
while error > tor and ep_step < (ep_step_initial + step_all):
    time_1 = time.time()
    ep_step += 1
    if ep_step % 15 == 0:
        LR_A *= 0.995
        Lamb_B *= 1.
        Lamb_B_ *= 1.
        Lamb_I *= 0.995
        Lamb_I_ *= 0.995
        if LR_A < 2e-6:
            LR_A = 2e-6
        if Lamb_B < 1.:
            Lamb_B = 1.
        if Lamb_B_ < 1.:
            Lamb_B_ = 1.
        if Lamb_I < 1.:
            Lamb_I = 1.
        if Lamb_I_ < 1.:
            Lamb_I_ = 1.

    a_loss, aloss_b, aloss_b_, aloss_i, aloss_i_, aloss_e = pg.learn_a(LR_A, Lamb_B, Lamb_B_, Lamb_I, Lamb_I_, S_all)

    a_b_q, a_b_q_2, a_i_q, a_i_q_2, a_e_q = pg.choose_action(S_all)

    R_b = np.mean(np.square(a_b_q[:, 0]) + np.square(a_b_q[:, 1]) + np.square(a_b_q[:, 2])) + np.mean(
        np.square(a_b_q[:, 3] + a_b_q[:, 9]))
    R_b_ = np.mean(np.square(a_b_q_2[:, 0] - 1.) + np.square(a_b_q_2[:, 1]) + np.square(a_b_q_2[:, 2])) + np.mean(
        np.square(a_b_q_2[:, 3] + a_b_q_2[:, 9]))

    R_i = np.mean(np.square(a_i_q[:, 1])) + np.mean(np.square(a_i_q[:, 2])) + np.mean(
        np.square(a_i_q[:, 3] + a_i_q[:, 9]))
    R_i_ = np.mean(np.square(a_i_q_2[:, 2])) + np.mean(np.square(a_i_q_2[:, 3] + a_i_q_2[:, 9]))

    R_e_u = a_e_q[:, 0][:, np.newaxis] * a_e_q[:, 3][:, np.newaxis] + a_e_q[:, 1][:, np.newaxis] * a_e_q[:, 5][:, np.newaxis] + \
            1. / rho * a_e_q[:, 11][:, np.newaxis] - miu / rho * (a_e_q[:, 4][:, np.newaxis] + a_e_q[:, 6][:, np.newaxis])
    R_e_v = a_e_q[:, 0][:, np.newaxis] * a_e_q[:, 7][:, np.newaxis] + a_e_q[:, 1][:, np.newaxis] * a_e_q[:, 9][:, np.newaxis] + \
            1. / rho * a_e_q[:, 12][:, np.newaxis] - miu / rho * (a_e_q[:, 8][:, np.newaxis] + a_e_q[:, 10][:, np.newaxis])

    R_e = np.mean(np.square(R_e_u) + np.square(R_e_v)) + np.mean(
        np.square(a_e_q[:, 3][:, np.newaxis] + a_e_q[:, 9][:, np.newaxis]))

    error = R_b + R_b_ + R_i + R_i_ + R_e

    print('')
    print('[Iteration_step] %d ************************** '
          '[Iteration_time] %10.8f' % (ep_step, time.time() - time_1))
    print('[error_b] %10.8f  [error_b_] %10.8f  [error_i] %10.8f  [error_i_] %10.8f  [error_e] %10.8f  '
          '[error] %10.8f' % (R_b, R_b_, R_i, R_i_, R_e, error))
    print('[aloss_b] %10.8f  [aloss_b_] %10.8f  [aloss_i] %10.8f  [aloss_i_] %10.8f  [aloss_e] '
          '%10.8f  [a_loss] %10.8f' % (aloss_b, aloss_b_, aloss_i, aloss_i_, aloss_e, a_loss))
    print('')

    err.append(error)
    err_b.append(np.mean(R_b))
    err_b_.append(np.mean(R_b_))
    err_i.append(np.mean(R_i))
    err_i_.append(np.mean(R_i_))
    err_e.append(np.mean(R_e))
    if ep_step % 100 == 0:
        S_Col = sio.loadmat('./data/S_Col.mat')
        S_Col = S_Col['S_Col']
        aa = pg.sess.run(pg.a, {pg.S: S_Col})

        Ua = np.concatenate([Ua, aa], axis=1)

sio.savemat('err.mat', {'err': err})
sio.savemat('err_b.mat', {'err_b': err_b})
sio.savemat('err_b_.mat', {'err_b_': err_b_})
sio.savemat('err_i.mat', {'err_i': err_i})
sio.savemat('err_i_.mat', {'err_i_': err_i_})
sio.savemat('err_e.mat', {'err_e': err_e})

print('Running time: ', time.time() - t1)

saver.save(pg.sess, './model/ddpgnet_result.cpkt')
sio.savemat('Ua.mat', {'Ua': Ua})
