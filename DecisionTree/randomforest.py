# encoding=utf-8
# 1.choose N examples randomly
# 2.choose m variables (M totally): round(log(M)/log(2)+1);
# 3.repeat 1-2 for k times, and create k decision trees
# 4.out of bag errors
import random
import math
import json
from ID3 import loadFile, createTree, classify


def bagNExample(dataset, labels, K, M):
    N = len(dataset) - 1
    m = int(round(math.log(M) / math.log(2) + 1))
    T = []
    L = []
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
            vec = []
            for j in varset:
                vec.append(dataset[i][j])
            vec.append(dataset[i][-1])
            Tk.append(vec)
        T.append(Tk)
        Lk = [labels[j] for j in varset]
        L.append(Lk)
    return T, L


def buildForest(dataset, labels, K, M):
    T, L = bagNExample(dataset, labels, K=K, M=M)
    forest = []
    for k in range(K):
        dtree = createTree(T[k], L[k])
        # if each in Tk is identical, it just return Yes. Just skip it
        if isinstance(dtree, dict):
            forest.append(dtree)
        # print '----------'
    return forest


def voteMajority(forest, labels, testVec):
    result = {}
    for i, dtree in enumerate(forest):
        try:
            cls = classify(dtree, labels, testVec)
        except Exception, e:
            # test vector may has some values not included in dtree
            # haven't come up a good solution, just mark it
            print 'id:', i, '--', e
            # print json.dumps(dtree, indent=4)
        else:
            if cls not in result:
                result[cls] = 1
            else:
                result[cls] += 1
    print result
    return


def main():
    dataset = loadFile('trainset.txt')
    labels = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    testVec = "Rain Mild High Weak".split(" ")
    forest = buildForest(dataset, labels, K=100, M=4)
    # print json.dumps(forest, indent=4)
    voteMajority(forest, labels, testVec)
    # print forest

if __name__ == '__main__':
    print "start"
    main()
