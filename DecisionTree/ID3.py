#encoding:utf-8
__author__ = 'wangjiewen'

import numpy
import operator
import json
from math import log

class ID3(object):
    def __init__(self):
        self.matrix = []
        self.labels = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        # self.loadFile('trainset.txt')

    def loadFile(self, filename):
        """
        加载数据文件
        :param filename: 文件路径
        :return: void
        """
        fr = open(filename)
        lines = fr.readlines()

        for lineStr in lines:
            lineArr = lineStr.strip().split(' ')
            self.matrix.append(lineArr)

    def calcEntropy(self, dataSet):
        """
        计算熵 entropy = -∑p(x) * log p(x)
        :param dataSet: 数据矩阵，最后一列为类标号
        :return: float 熵值
        """
        nums = len(dataSet)
        labelsCount = {}

        #统计每个类别的数量
        for featVec in dataSet:
            classLabel = featVec[-1]
            if classLabel not in labelsCount.keys():
                labelsCount[classLabel] = 0
            labelsCount[classLabel] += 1

        entropy = 0.0
        for label in labelsCount:
            prob = float(labelsCount[label]) / nums
            entropy -= prob * log(prob, 2)
        return entropy


    def splitDataSet(self, dataSet, col):
        """
        将dataSet按照第col列分割，返回第col列值为value的数据
        :param dataSet: []
        :param col: 分割的列
        :return: {} col列所取值的对象数组
        """
        # result = []
        # for featVec in dataSet:
        #     if featVec[col] == value:
        #         tmpLeft = featVec[:col]
        #         tmpLeft.extend(featVec[col + 1:])
        #         result.append(tmpLeft)
        result = {}
        for featVec in dataSet:
            # tmpLine = featVec[:col]
            # tmpLine.extend(featVec[col + 1:])
            tmpLine = featVec[:]

            key = featVec[col]
            if key not in result.keys():
                tmpArr = []
                tmpArr.append(tmpLine)
                result[key] = tmpArr
            else:
                tmpArr = result[key]
                tmpArr.append(tmpLine)


        return result

    def selectMaxGainCol(self, dataSet):
        """
        选择信息最大信息增益的列标号
        :param dataSet:
        :return:
        """
        numsOfCol = len(dataSet[0]) - 1 #数据集的列数，最后一列为类标号
        numsOfData = len(dataSet) #数据集的条目数量

        maxGain = 0.0 #最大的信息增益
        maxFeatCol = -1 #最大信息增益对应的列

        #Entropy(S)
        entropyS = self.calcEntropy(dataSet)

        for col in range(0, numsOfCol):
            featDict = self.splitDataSet(dataSet, col)

            tmpGain = entropyS
            for key in featDict.keys():
                featArr = featDict[key]
                entropyFeat = self.calcEntropy(featArr)
                delta = (len(featArr) / float(numsOfData)) * entropyFeat
                tmpGain -= delta

            # print "Gain(", self.labels[col], ") =", tmpGain
            if tmpGain > maxGain:
                maxGain = tmpGain
                maxFeatCol = col
        return maxFeatCol


    def majorityCnt(self, classList):
        classCount = {}
        for key in classList:
            if key not in classCount.keys():
                classCount[key] = 0
            classCount[key] += 1
        sortedClass = sorted(classCount, key=operator.getitem(1), reverse=True)
        return sortedClass[0][0]

    def createTree(self):
        initCol = self.selectMaxGainCol(self.matrix)
        root = {self.labels[initCol]: {}}

        #节点栈，保存当前访问的节点
        nodeStack = []
        nodeStack.append(root[self.labels[initCol]])

        #类标号栈，保存当前进行划分的类标号
        colStack = []
        colStack.append(initCol)

        #数据站，保存当前需要被划分的数据，和类标号一一对应
        dataStack = []
        dataStack.append(self.matrix)

        while(len(dataStack) > 0):
            dataSet = dataStack.pop()
            col = colStack.pop()
            pCur = nodeStack.pop() #指向当前节点的指针

            #按属性进行划分后的数据字典
            splitDict = self.splitDataSet(dataSet, col)

            for key in splitDict:
                data = splitDict[key]

                #如果全部属于正类或负类，则标记其类别，代表已经划分完成
                classSet = set(example[-1] for example in data)
                if len(classSet) == 1:
                    endLabel = classSet.pop()
                    label = self.labels[col]

                    #分情况讨论，当节点具有分支的时候，
                    if label in pCur.keys():
                        pCur[label][key] = endLabel
                    else:
                        pCur[key] = endLabel
                    continue

                #如果属性的还可以继续划分，则将该节点加入对应的栈中
                #因为最后一行为类标号，因此要>1
                if len(data[0]) > 1:
                    tmpMaxCol = self.selectMaxGainCol(data)
                    label = self.labels[tmpMaxCol]
                    pCur[key] = {}
                    pCur[key][label] = {}
                    dataStack.append(data)
                    colStack.append(tmpMaxCol)
                    nodeStack.append(pCur[key])

        print json.dumps(root, indent=4)

        return root



    def test(self):
        dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        self.matrix = dataSet
        self.labels = labels

        # print dataSet


model = ID3()
model.test()
model.createTree()