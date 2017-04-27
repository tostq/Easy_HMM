# -*- coding:utf-8 -*-
# By tostq <tostq216@163.com>
# 博客: blog.csdn.net/tostq
from hmmlearn.hmm import MultinomialHMM
import numpy as np
import hmm

dice_num = 3
x_num = 8
dice_hmm = hmm.DiscreteHMM(3, 8)
dice_hmm.start_prob = np.ones(3)/3.0
dice_hmm.transmat_prob = np.ones((3,3))/3.0
dice_hmm.emission_prob = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
# 归一化
dice_hmm.emission_prob = dice_hmm.emission_prob / np.repeat(np.sum(dice_hmm.emission_prob, 1),8).reshape((3,8))

dice_hmm.trained = True

X = np.array([[1],[6],[3],[5],[2],[7],[3],[5],[2],[4],[3],[6],[1],[5],[4]])
Z = dice_hmm.decode(X) # 问题A
logprob = dice_hmm.X_prob(X) # 问题B

# 问题C
x_next = np.zeros((x_num,dice_num))
for i in range(x_num):
    c = np.array([i])
    x_next[i] = dice_hmm.predict(X, i)

print "state: ", Z
print "logprob: ", logprob
print "prob of x_next: ", x_next

