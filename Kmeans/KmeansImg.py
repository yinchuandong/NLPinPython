import numpy as np
from PIL import Image
from sklearn import metrics
from sklearn.cluster import KMeans


def main():
    im = Image.open('1.jpg')
    mat = np.array(im)
    print np.shape(mat)

    return

def testKMeans():
    im = Image.open('1.jpg')
    data = im.getdata()
    km = KMeans()
    rt = km.fit(data)
    print rt.labels_[200:500]
    return

def kmeans_in_matrix():
    # 5 points, 2 feature
    X = [
        [1, -1, 0, 0, 0],
        [0, 0, 2, 3, 4]]
    A = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ]
    # A = [
    #     [1, 0, 0],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ]
    X = np.mat(X)
    A = np.mat(A)
    n1 = np.ones((5, 1))
    n1 = np.mat(n1)
    t = (n1.T * A)
    t = np.diag(t.A1)
    t = np.mat(t)
    C = X * A * t.I
    print C
    D = X - C * A.T
    print D
    r = np.power(D, 2)
    print np.sum(r)

    return

if __name__ == '__main__':
    # main()
    # testKMeans()