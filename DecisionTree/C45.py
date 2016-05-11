# encoding:utf-8
__author__ = 'yinchuandong'

import json
from math import log

from ID3 import calcEntropy, splitDataSet, loadFile


def selectMaxGaniRatio(dataSet):
    numsOfCol = len(dataSet[0]) - 1
    numsOfData = len(dataSet)

    maxGainRatio = 0.0
    maxFeatCol = 0
    entropyS = calcEntropy(dataSet)
    for col in range(numsOfCol):
        subData = splitDataSet(dataSet, col)
        gain = entropyS
        splitInfo = 0.0
        for key in subData:
            entropy = calcEntropy(subData[key])
            prob = len(subData[key]) / float(numsOfData)
            gain -= prob * entropy
            splitInfo -= prob * log(prob)
        gainRatio = gain / splitInfo
        if gainRatio > maxGainRatio:
            maxGainRatio = gainRatio
            maxFeatCol = col
    return maxFeatCol


def createTree(dataSet, labels):
    classList = [featVec[-1] for featVec in dataSet]
    if len(set(classList)) == 1:
        return classList[0]
    bestCol = selectMaxGaniRatio(dataSet)
    bestColLabel = labels[bestCol]
    tree = {}
    tree[bestColLabel] = {}
    del(labels[bestCol])  # del after spliting on it
    featVals = [featVec[bestCol] for featVec in dataSet]
    uniqueVals = set(featVals)
    subData = splitDataSet(dataSet, bestCol)
    for val in uniqueVals:
        subtree = createTree(subData[val], labels[:])
        tree[bestColLabel][val] = subtree
    return tree


def main():
    dataSet = loadFile('trainset.txt')
    labels = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    tree = createTree(dataSet, labels)
    print json.dumps(tree, indent=4)
    return

if __name__ == '__main__':
    main()
