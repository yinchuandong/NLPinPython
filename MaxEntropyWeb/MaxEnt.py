__author__ = 'wangjiewen'

from collections import defaultdict
import math

class MaxEnt(object):
    def __init__(self):
        #
        self.feats = defaultdict(int)
        self.trainset = []
        self.labels = set()

    #load train file
    def load_data(self,file):
        for line in open(file):
            fields = line.strip().split()
            # at least two columns
            if len(fields) < 2: continue
            # the first column is label
            label = fields[0]
            self.labels.add(label)
            for f in set(fields[1:]):
                # (label,f) tuple is feature
                self.feats[(label,f)] += 1
            self.trainset.append(fields)

    def _initparams(self):
        self.size = len(self.trainset)
        # M param for GIS training algorithm
        self.M = max([len(record)-1 for record in self.trainset])
        self.ep_ = [0.0]*len(self.feats)
        for i,f in enumerate(self.feats):
            # calculate feature expectation on empirical distribution
            self.ep_[i] = float(self.feats[f])/float(self.size)
            # each feature function correspond to id
            self.feats[f] = i
        # init weight for each feature
        self.w = [0.0]*len(self.feats)
        self.lastw = self.w

    def probwgt(self,features,label):
        wgt = 0.0
        for f in features:
            if (label,f) in self.feats:
                wgt += self.w[self.feats[(label,f)]]
        return math.exp(wgt)

    def calprob(self,features):
        wgts = [(self.probwgt(features, l),l) for l in self.labels]
        Z = sum([ w for w,l in wgts])
        prob = [ (w/Z,l) for w,l in wgts]
        return prob

    """
    calculate feature expectation on model distribution
    """
    def Ep(self):
        ep = [0.0]*len(self.feats)
        for record in self.trainset:
            features = record[1:]
            # calculate p(y|x)
            prob = self.calprob(features)
            for f in features:
                for w,l in prob:
                    # only focus on features from training data.
                    if (l,f) in self.feats:
                        # get feature id
                        idx = self.feats[(l,f)]
                        # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                        ep[idx] += w * (1.0/self.size)
        return ep

    def _convergence(self,lastw,w):
        for w1,w2 in zip(lastw,w):
            if abs(w1-w2) >= 0.01:
                return False
        return True


    def train(self, max_iter =1000):
        self._initparams()
        for i in range(max_iter):
            print 'iter %d ...'%(i+1)
            # calculate feature expectation on model distribution
            self.ep = self.Ep()
            self.lastw = self.w[:]
            for i,w in enumerate(self.w):
                delta = 1.0/self.M * math.log(self.ep_[i]/self.ep[i])
                # update w
                self.w[i] += delta
            print self.w
            # test if the algorithm is convergence
            if self._convergence(self.lastw,self.w):
                break


    def predict(self,input):
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)
        return prob