#encoding:utf-8
__author__ = 'wangjiewen'

from collections import defaultdict
import math

class MaxEntropy(object):

    def __init__(self):
        self.trainList = []
        self.featureDict = defaultdict(int)
        self.indexDict = defaultdict(int)
        self.labels = set()

        self.ep = []
        self.ep_ = []
        self.lambdaNew = []
        self.lambdaOld = []
        self.epsilon = 0.01
        self.maxIterator = 1000
        self.N = 0
        self.C = 0


    def loadData(self, filepath):
        files = open(filepath)
        for line in files:
            fields = line.strip().split(' ')
            label = fields[0]
            self.labels.add(label)

            for feature in fields[1:]:
                self.featureDict[(label, feature)] += 1

            self.trainList.append(fields)

    def initParams(self):
        self.N = len(self.trainList)
        self.C = max([len(record) - 1 for record in self.trainList])

        self.ep_ = [0.0] * len(self.featureDict)
        for i, feature in enumerate(self.featureDict):
            prob = float(self.featureDict[feature]) / float(self.N)
            self.ep_[i] = prob
            self.indexDict[feature] = i

        self.lambdaNew = [0.0] * len(self.featureDict)
        self.lambdaOld = self.lambdaNew


    def zFunc(self, features, label):
        """
        计算Z里面的 exp(∑ λi*fi(a,b))

        :param features: string[] 特征的数组
        :param label: string
        :return: double
        """
        weight = 0.0
        for f in features:
            if (label, f) in self.featureDict:
                index = self.indexDict[(label, f)]
                weight += self.lambdaNew[index]

        return math.exp(weight)


    def pFunc(self, features, label):
        Z = 0.0
        for l in self.labels:
            Z += self.zFunc(features, l)

        prob = (1.0 / Z) * self.zFunc(features,label)
        return prob

    def calcEp(self):
        ep = [0.0] * len(self.featureDict)

        for record in self.trainList:
            features = record[1:]
            for label in self.labels:
                prob = self.pFunc(features, label)

                # ∑ p(a) * p(b|a) * f(a,b), p(a) = 1/N
                for f in features:
                    if (label, f) in self.featureDict:
                        index = self.indexDict[(label, f)]
                        ep[index] += (1.0 / self.N) * prob
        return ep

    def isConvergent(self, lambdaNew, lambdaOld):
        for l1, l2 in zip(lambdaNew, lambdaOld):
            if abs(l1 - l2) >= self.epsilon:
                return False
        return True

    def train(self):
        self.initParams()

        for k in range(0, self.maxIterator):
            self.ep = self.calcEp()
            self.lambdaOld = self.lambdaNew[:]

            for i, l1 in enumerate(self.lambdaNew):
                delta = 1.0 / self.C * math.log(self.ep_[i] / self.ep[i])
                self.lambdaNew[i] += delta

            if self.isConvergent(self.lambdaNew, self.lambdaOld):
                break


    def predict(self, features):
        for label in self.labels:
            prob = self.pFunc(features, label)
            print (prob, label)


model = MaxEntropy()
model.loadData('train.txt')
model.train()

model.predict(['Sunny', 'Happy'])
