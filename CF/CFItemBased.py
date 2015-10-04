# encoding:utf-8

import numpy as np
import numpy.linalg as la


def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(
            dataMat[:, item].A > 0,
            dataMat[:, j].A > 0
        ))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item],
                                 dataMat[overLap, j])
        print 'the item %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def main():
    # dataMat = np.mat(loadExData())
    # dataMat[0, 1] = dataMat[0, 0] = dataMat[1, 0] = dataMat[2, 0] = 4
    # dataMat[3, 3] = 2
    dataMat = np.mat([
        [4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0]])

    inA = dataMat[:, 0]
    inB = dataMat[:, 4]
    # print ecludSim(inA, inB)
    # print pearsSim(inA, inB)
    # print cosSim(inA, inB)

    print standEst(dataMat, 2, cosSim, 2)

    return

if __name__ == '__main__':
    main()
