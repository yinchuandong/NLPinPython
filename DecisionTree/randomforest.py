# encoding=utf-8
# 1.choose N examples randomly
# 2.choose m variables (M totally): round(log(M)/log(2)+1);
# 3.repeat 1-2 for k times, and create k decision trees
# 4.out of bag errors
import numpy as np
import matplotlib.pyplot as pl
import random
import math
from ID3 import loadFile, createTree, classify


def bagNExample(dataset, K, M):
    N = len(dataset) - 1
    m = int(round(math.log(M) / math.log(2) + 1))
    ret = []
    for k in range(K):
        # bag of example
        subset = []
        for i in range(N):
            randId = random.randint(0, N - 1)
            subset.append(randId)
        # bag of variable
        varset = []
        for j in range(m):
            randId = random.randint(0, M - 1)
            while(randId in varset):
                randId = random.randint(0, M - 1)
            varset.append(randId)
        varset = sorted(varset)

        Tk = []
        for i in subset:
            row = []
            for j in varset:
                row.append(dataset[i][j])
            row.append(dataset[i][-1])
            Tk.append(row)
        ret.append(Tk)
    return ret


def main():
    dataset = loadFile('trainset.txt')
    labels = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    T = bagNExample(dataset, K=3, M=4)
    print T
    # tree = createTree(dataset, labels)

if __name__ == '__main__':
    print "start"
    main()