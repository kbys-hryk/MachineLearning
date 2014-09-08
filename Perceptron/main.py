__author__ = 'kbysPC'
# coding:utf-8

import numpy
import theano
import theano.tensor as T


"""
変数宣言
"""
input = []
output = []
node_n = []
depth = 0
weight = []

"""
教師データ,パラメータをセットする関数
"""

def and_functiopn_data(self):
    self.input = [
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0]
    ]

    self.output = [
        0,
        1,
        0,
        1
    ]
    self.node_n = [2,1]
    self.depth = 1

    weight = numpy.zeros()


if __name__ == "__main__":
    print 0