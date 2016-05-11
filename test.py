import numpy as np


def main():
    D = [0, 0, 1, 1]
    X = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ]
    C = [
        [2, 3],
        [6, 7]
    ]
    D = np.mat(D).T
    X = np.mat(X)
    C = np.mat(C)
    print C[D]
    # print X[(D[:, 0].A == 1)]
    # print np.nonzero(D[:, 0].A == 0)
    # print X[np.nonzero(D[:, 0].A == 1)[0]]
    return



if __name__ == '__main__':
    main()