__author__ = 'wangjiewen'

from numpy import *

dataSet = mat([[0, 2], [0, 1], [1, 1], [2, 1], [3, 2]])

best = mat([[3, 4], [5, 6], [7, 8]])

dataSet[nonzero(dataSet[:, 1].A == 1)[0], :] = best

print dataSet
