import numpy as np


def loadData():
    # mat = [
    #     [13, 16, 12, 11],
    #     [15, 17, 12, 12],
    #     [14, 14, 13, 13],
    #     [13, 10, 10, 11]
    # ]
    mat = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]
    return mat

def test():
    from munkres import Munkres, print_matrix

    matrix = loadData()

    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total cost: %d' % total



if __name__ == '__main__':
    test()
