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


if __name__ == '__main__':
    # main()
    testKMeans()