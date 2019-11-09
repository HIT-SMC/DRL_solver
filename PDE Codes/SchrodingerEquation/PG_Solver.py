"""
DRL Solver for Schrodinger Equation
ut=0.5i*uxx+i*|u|^2*u
Author: Shiyin Wei, Xiaowei Jin, Hui Li*
Date: April, 2018
"""

import tensorflow as tf
import numpy as np
import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# ********** GPU ratio ********** #
GR = 0.9

# ********** Points sampling ********** #
Num = 999
Num_B = 10
Num_I = 100
S_x_ = np.linspace(-5., 5., Num + 2)
S_x = S_x_[1:-1][:, np.newaxis]
deltaT = 0.01

# ********** Hyper parameters ********** #
LR_A_ini = 5e-4  # Initial learning rate for actor
Lamb_I = 1.  # Coefficient for the initial conditions
Lamb_B = 1.  # Coefficient for the boundary conditions
tor = 5e-5
forward_T_n = 10


# ********** PG ********** #
class PG(object):
    def __init__(self, a_dim, s_dim):
        self.pointer = 0
        self.lr = tf.placeholder(tf.float32, None, 'LR')
        self.Lamb_B = tf.placeholder(tf.float32, None, 'L_B')
        self.Lamb_I = tf.placeholder(tf.float32, None, 'L_I')
        # b_num = tf.placeholder(tf.int32, None, 'b_num')
        b_num = 100*Num_B
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = GR
        self.sess = tf.Session(config=config)
        self.on_train = tf.constant(True, dtype=bool)
        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')

        self.a = self._build_a(self.S)
        self.normal_dist_re = tf.distributions.Normal(self.a[:, 0], self.a[:, 8])
        self.normal_dist_prob_re = tf.expand_dims(self.normal_dist_re.prob(self.a[:, 0]), 1)
        self.normal_dist_im = tf.distributions.Normal(self.a[:, 4], self.a[:, 9])
        self.normal_dist_prob_im = tf.expand_dims(self.normal_dist_im.prob(self.a[:, 4]), 1)

        self.a_i = self.a[:Num_I, :]
        self.pi_i_re = self.normal_dist_prob_re[:Num_I, :]
        self.pi_i_im = self.normal_dist_prob_im[:Num_I, :]

        self.a_b = self.a[Num_I:Num_I + b_num*2, :]
        self.pi_b_re_1 = self.normal_dist_prob_re[Num_I:Num_I + b_num, :]
        self.pi_b_re_2 = self.normal_dist_prob_re[Num_I + b_num:Num_I + b_num*2, :]
        self.pi_b_re = 0.5*(self.pi_b_re_1 + self.pi_b_re_2)
        self.pi_b_im_1 = self.normal_dist_prob_im[Num_I:Num_I + b_num, :]
        self.pi_b_im_2 = self.normal_dist_prob_im[Num_I + b_num:Num_I + b_num*2, :]
        self.pi_b_im = 0.5 * (self.pi_b_im_1 + self.pi_b_im_2)
        self.a_e = self.a[Num_I + b_num*2:, :]
        self.pi_e_re = self.normal_dist_prob_re[Num_I + b_num*2:, :]
        self.pi_e_im = self.normal_dist_prob_im[Num_I + b_num*2:, :]

        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')

        R_real_i = self.a_i[:, 0:1] - 2./tf.cosh(self.S[:Num_I, 1:])
        R_imag_i = self.a_i[:, 4:5]

        R_real_b = self.a_b[:b_num, 0:1] - self.a_b[b_num:b_num*2, 0:1]
        R_imag_b = self.a_b[:b_num, 4:5] - self.a_b[b_num:b_num*2, 4:5]
        R_real_b_x = (self.a_b[:b_num, 2:3] - self.a_b[b_num:b_num*2, 2:3])
        R_imag_b_x = (self.a_b[:b_num, 6:7] - self.a_b[b_num:b_num*2, 6:7])

        R_real_e = self.a_e[:, 1:2] + 0.5 * self.a_e[:, 7:8] + \
                   (self.a_e[:, 0:1] ** 2. + self.a_e[:, 4:5] ** 2.) * self.a_e[:, 4:5]
        R_imag_e = -self.a_e[:, 5:6] + 0.5 * self.a_e[:, 3:4] + \
                   (self.a_e[:, 0:1] ** 2. + self.a_e[:, 4:5] ** 2.) * self.a_e[:, 0:1]

        self.r_real_i = tf.reduce_mean(tf.square(R_real_i))
        self.r_imag_i = tf.reduce_mean(tf.square(R_imag_i))

        self.r_real_b_1 = tf.reduce_mean(tf.square(R_real_b))
        self.r_real_b_2 = tf.reduce_mean(tf.square(R_real_b_x))
        self.r_real_e = tf.reduce_mean(tf.square(R_real_e))

        self.r_imag_b_1 = tf.reduce_mean(tf.square(R_imag_b))
        self.r_imag_b_2 = tf.reduce_mean(tf.square(R_imag_b_x))
        self.r_imag_e = tf.reduce_mean(tf.square(R_imag_e))
        self.r = self.r_real_i + self.r_imag_i + self.r_real_b_1 + self.r_real_b_2 + \
                 self.r_real_e + self.r_imag_b_1 + self.r_imag_b_2 + self.r_imag_e

        self.q_real_i = tf.reduce_mean(tf.square(R_real_i)*self.pi_i_re)*self.Lamb_I
        self.q_imag_i = tf.reduce_mean(tf.square(R_imag_i)*self.pi_i_im)*self.Lamb_I

        self.q_real_b = tf.reduce_mean(tf.square(R_real_b)*self.pi_b_re + tf.square(R_real_b_x)*self.pi_b_re)*self.Lamb_B
        self.q_real_e = tf.reduce_mean(tf.square(R_real_e)*self.pi_e_re)

        self.q_imag_b = tf.reduce_mean(tf.square(R_imag_b)*self.pi_b_im + tf.square(R_imag_b_x)*self.pi_b_im)*self.Lamb_B
        self.q_imag_e = tf.reduce_mean(tf.square(R_imag_e)*self.pi_e_im)

        self.a_loss = self.q_real_i + self.q_imag_i + self.q_real_b + self.q_real_e + self.q_imag_b + self.q_imag_e
        self.atrain = tf.train.AdamOptimizer(self.lr).minimize(self.a_loss, var_list=self.a_params)

        self.sess.run(tf.global_variables_initializer())

    def learn_a(self, learn_rate, lamb_b, lamb_i, bs):
        self.sess.run(self.atrain, {self.S: bs, self.lr: learn_rate, self.Lamb_B: lamb_b, self.Lamb_I: lamb_i})
        aloss, erro, aloss_real_i, a_loss_imag_i, \
        aloss_real_b_1, aloss_real_b_2, aloss_real_e, \
        aloss_imag_b_1, aloss_imag_b_2, aloss_imag_e = \
            self.sess.run([self.a_loss, self.r, self.r_real_i, self.r_imag_i,
                           self.r_real_b_1, self.r_real_b_2, self.r_real_e,
                           self.r_imag_b_1, self.r_imag_b_2, self.r_imag_e],
                          {self.S: bs, self.lr: learn_rate, self.Lamb_B: lamb_b, self.Lamb_I: lamb_i})
        return aloss, erro, aloss_real_i, a_loss_imag_i, \
               aloss_real_b_1, aloss_real_b_2, aloss_real_e, aloss_imag_b_1, aloss_imag_b_2, aloss_imag_e

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
            sigma = tf.nn.sigmoid(tf.layers.dense(l7, 2, name='a_sigma', trainable=trainable))
            u_ = tf.layers.dense(l7, 2, name='a_action', trainable=trainable)
            # ur = s[:, 0:1]*u_[:, 0:1] + 2./tf.cosh(s[:, 1:])
            ur = u_[:, 0:1]
            # ui = s[:, 0:1]*u_[:, 1:2]
            ui = u_[:, 1:2]
            hr_t = tf.expand_dims(tf.gradients(ur[:, 0], s)[0][:, 0], 1)

            hr_x = tf.expand_dims(tf.gradients(ur[:, 0], s)[0][:, 1], 1)
            hr_xx = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(ur[:, 0], s)[0][:, 1], 1),
                                                s)[0][:, 1], 1)

            hi_t = tf.expand_dims(tf.gradients(ui[:, 0], s)[0][:, 0], 1)

            hi_x = tf.expand_dims(tf.gradients(ui[:, 0], s)[0][:, 1], 1)
            hi_xx = tf.expand_dims(tf.gradients(tf.expand_dims(tf.gradients(ui[:, 0], s)[0][:, 1], 1),
                                                s)[0][:, 1], 1)

            action = tf.concat([ur, hr_t, hr_x, hr_xx, ui, hi_t, hi_x, hi_xx, sigma], axis=1)
            return action


# ********** TRAINING ********** #
s_dim_ = 2
a_dim_ = 8

pg = PG(a_dim_, s_dim_)

t1 = time.time()

sess = tf.Session()
saver = tf.train.Saver()

start = time.time()

time_0 = time.time()
init_real_now = 2./np.cosh(S_x)
init_imag_now = np.zeros([Num, 1], dtype=np.float32)
err = []
Col_real = init_real_now
Col_imag = init_imag_now
IteNo = []

for ti in range(157):
    back_num = forward_T_n*Num
    if back_num < 30000:
        forward_T_n += 1
    tn = (ti + 1)*deltaT
    S_t_I = np.zeros([Num_I, 1], dtype=np.float32)
    S_x_I = np.reshape(np.random.rand(Num_I), [Num_I, 1]) * 10. - 5.
    S_I = np.concatenate([S_t_I, S_x_I], axis=1)

    S_t_B_1 = np.reshape(np.random.rand((100 - 1) * Num_B), [(100 - 1) * Num_B, 1]) * tn
    S_t_B_2 = tn * np.ones([Num_B, 1], dtype=np.float32)
    S_t_B = np.concatenate([S_t_B_1, S_t_B_2, S_t_B_1, S_t_B_2], axis=0)
    S_x_B_plus = 5. * np.ones([100*Num_B, 1], dtype=np.float32)
    S_x_B_minus = -5. * np.ones([100*Num_B, 1], dtype=np.float32)
    S_x_B = np.concatenate([S_x_B_plus, S_x_B_minus], axis=0)
    S_all_B = np.concatenate([S_t_B, S_x_B], axis=1)

    S_t_now_1 = np.reshape(np.random.rand((forward_T_n - 1) * Num), [(forward_T_n - 1) * Num, 1]) * tn
    S_t_now_2 = tn * np.ones([Num, 1], dtype=np.float32)
    S_x_1 = np.reshape(np.random.rand((forward_T_n - 1) * Num), [(forward_T_n - 1) * Num, 1]) * 10. - 5.
    S_now_1_all = np.concatenate([S_t_now_1, S_x_1], axis=1)
    S_now_2_all = np.concatenate([S_t_now_2, S_x], axis=1)
    S_all = np.concatenate([S_I, S_all_B, S_now_1_all, S_now_2_all], axis=0)

    LR_A = LR_A_ini
    error = 1e4
    aloss_real_i = 1e4
    a_loss_imag_i = 1e4
    aloss_real_b_1 = 1e4
    aloss_real_b_2 = 1e4
    aloss_real_e = 1e4
    aloss_imag_b_1 = 1e4
    aloss_imag_b_2 = 1e4
    aloss_imag_e = 1e4
    ep_step = 0
    while error > tor:
        time_1 = time.time()
        ep_step += 1
        if ep_step % 100 == 0:
            LR_A *= 0.95
            if LR_A < 3e-6:
                LR_A = 3e-6

        a_loss, error, aloss_real_i, a_loss_imag_i, \
        aloss_real_b_1, aloss_real_b_2, aloss_real_e, aloss_imag_b_1, aloss_imag_b_2, aloss_imag_e = \
            pg.learn_a(LR_A, Lamb_B, Lamb_I, S_all)

        print('')
        print('[Time_step] %d [Iteration_step] %d ************************** '
              '[Iteration_time] %10.8f' % (ti + 1, ep_step, time.time() - time_1))
        print('[a_loss] %10.8f' % a_loss)
        print('[error] %10.8f' % error)
        print('[error_real_i] %10.8f [error_imag_i] %10.8f' % (aloss_real_i, a_loss_imag_i))
        print('[err_real_b_1] %10.8f [err_real_b_2] %10.8f [err_real_e] %10.8f' % (aloss_real_b_1, aloss_real_b_2, aloss_real_e))
        print('[err_imag_b_1] %10.8f [[err_imag_b_2] %10.8f [err_imag_e] %10.8f' % (aloss_imag_b_1, aloss_imag_b_2, aloss_imag_e))
        print('[Learning rate] %10.8f' % LR_A)
        print('')
        err.append([a_loss, error, aloss_real_i, a_loss_imag_i,
                    aloss_real_b_1, aloss_real_b_2, aloss_real_e, aloss_imag_b_1, aloss_imag_b_2, aloss_imag_e])
    IteNo.append(ep_step)
    init_real_now = pg.sess.run(pg.a[-Num:, 0:1], {pg.S: S_all})
    init_imag_now = pg.sess.run(pg.a[-Num:, 4:5], {pg.S: S_all})
    Col_real = np.concatenate([Col_real, init_real_now], axis=1)
    Col_imag = np.concatenate([Col_imag, init_imag_now], axis=1)

    np.savetxt('err.dat', err)
    np.savetxt('IteNo.dat', IteNo)
    np. savetxt('Col_real.dat', Col_real)
    np.savetxt('Col_imag.dat', Col_imag)

    print('Running time: ', time.time() - t1)

    saver.save(pg.sess, './model/dpnet_result.cpkt')
