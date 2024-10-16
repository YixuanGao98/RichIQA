import numpy as np

# calculate mean score for AVA dataset
def mean_score(scores):
    # si = np.arange(0.05, 1.05, 0.1)#得分KONIQ10K 起点是1，终点是6，步长为1。
    si = np.arange(0.10, 1.10, 0.20)#得分KONIQ10K 起点是1，终点是6，步长为1。

    mean = np.sum(scores * si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    # si = np.arange(0.05, 1.05, 0.1)#得分KONIQ10K 起点是1，终点是6，步长为1。
    si = np.arange(0.10, 1.10, 0.20)#得分KONIQ10K 起点是1，终点是6，步长为1。

    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std

    