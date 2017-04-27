# Easy_HMM
A easy HMM program written with Python, including the full codes of training, prediction and decoding.

# Introduction
- Simple algorithms and models to learn HMMs in pure Python
- Including two HMM models: HMM with Gaussian emissions, and HMM with multinomial (discrete) emissions
- Using unnitest to verify our performance with [hmmlearn](http://hmmlearn.readthedocs.io/en/latest/ "hmmlearn") . 
- Three examples: Dice problem, Chinese words segmentation and stock analysis.

# Code list
- hmm.py: hmm models file
- DiscreteHMM_test.py, GaussianHMM_test.py: test files
- Dice_01.py, Wordseg_02.py, Stock_03.py: example files
- RenMinData.txt_utf8: Chinese words segmentation datas

# 中文说明
参见个人博客：[http://blog.csdn.net/tostq/article/details/70846702](http://blog.csdn.net/tostq/article/details/70846702 "hmm")

里面具体剖析了HMM模型，这个代码也是上述系列博客的配套代码！