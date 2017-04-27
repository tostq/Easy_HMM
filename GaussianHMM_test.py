# -*- coding:utf-8 -*-
# 测试高斯概率的隐马尔科夫链模型
# 引入一个经典的HMM库 hmmlearn，作为参照组
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
import unittest
import hmm
import hmmlearn.hmm
import numpy as np
from math import sqrt

class ContrastHMM():
    def __init__(self, n_state, n_feature):
        self.module = hmmlearn.hmm.GaussianHMM(n_components=n_state,covariance_type="full")
        # 初始概率
        self.module.startprob_ = np.random.random(n_state)
        self.module.startprob_ = self.module.startprob_ / np.sum(self.module.startprob_)
        # 转换概率
        self.module.transmat_ = np.random.random((n_state,n_state))
        self.module.transmat_ = self.module.transmat_ / np.repeat(np.sum(self.module.transmat_, 1),n_state).reshape((n_state,n_state))
        # 高斯发射概率
        self.module.means_ = np.random.random(size=(n_state,n_feature))*10
        self.module.covars_ = .5 * np.tile(np.identity(n_feature), (n_state, 1, 1))

# 计算平方误差
def s_error(A, B):
    return sqrt(np.sum((A-B)*(A-B)))/np.sum(B)

class GaussianHMM_Test(unittest.TestCase):

    def setUp(self):
        # 建立两个HMM，隐藏状态个数为4，X可能分布为10类
        n_state =4
        n_feature = 2  # 表示观测值的维度
        X_length = 1000
        n_batch = 200 # 批量数目
        self.n_batch = n_batch
        self.X_length = X_length
        self.test_hmm = hmm.GaussianHMM(n_state, n_feature)
        self.comp_hmm = ContrastHMM(n_state, n_feature)
        self.X, self.Z = self.comp_hmm.module.sample(self.X_length*10)
        self.test_hmm.train(self.X, self.Z)

    def test_train_batch(self):
        X = []
        Z = []
        for b in range(self.n_batch):
            b_X, b_Z = self.comp_hmm.module.sample(self.X_length)
            X.append(b_X)
            Z.append(b_Z)

        batch_hmm = hmm.GaussianHMM(self.test_hmm.n_state, self.test_hmm.x_size)
        batch_hmm.train_batch(X, Z)
        # 判断概率参数是否接近
        self.assertAlmostEqual(s_error(batch_hmm.start_prob, self.comp_hmm.module.startprob_), 0, 1)
        self.assertAlmostEqual(s_error(batch_hmm.transmat_prob, self.comp_hmm.module.transmat_), 0, 1)
        self.assertAlmostEqual(s_error(batch_hmm.emit_means, self.comp_hmm.module.means_), 0, 1)
        self.assertAlmostEqual(s_error(batch_hmm.emit_covars, self.comp_hmm.module.covars_), 0, 1)

    def test_train(self):
        # 判断概率参数是否接近
        # 单批量的初始概率一定是不准的
        # self.assertAlmostEqual(s_error(self.test_hmm.start_prob, self.comp_hmm.module.startprob_), 0, 1)
        self.assertAlmostEqual(s_error(self.test_hmm.transmat_prob, self.comp_hmm.module.transmat_), 0, 1)
        self.assertAlmostEqual(s_error(self.test_hmm.emit_means, self.comp_hmm.module.means_), 0, 1)
        self.assertAlmostEqual(s_error(self.test_hmm.emit_covars, self.comp_hmm.module.covars_), 0, 1)

    def test_X_prob(self):
        X,_ = self.comp_hmm.module.sample(self.X_length)
        prob_test = self.test_hmm.X_prob(X)
        prob_comp = self.comp_hmm.module.score(X)
        self.assertAlmostEqual(s_error(prob_test, prob_comp), 0, 1)

    def test_predict(self):
        X, _ = self.comp_hmm.module.sample(self.X_length)
        prob_next = self.test_hmm.predict(X,np.random.random(self.test_hmm.x_size))
        self.assertEqual(prob_next.shape,(self.test_hmm.n_state,))

    def test_decode(self):
        X,_ = self.comp_hmm.module.sample(self.X_length)
        test_decode = self.test_hmm.decode(X)
        _, comp_decode = self.comp_hmm.module.decode(X)
        self.assertAlmostEqual(s_error(test_decode, comp_decode), 0, 1)

if __name__ == '__main__':
    unittest.main()