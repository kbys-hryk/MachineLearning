__author__ = 'kbysPC'
# coding:utf-8
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

"""
Functionメソッドについて
実行しているコードの箇所
"""


def hoge():
    return 1

def functionTest():
    """
    Functionメソッドについて
    実行しているコードの箇所
    """
    x = T.lscalar("x")
    y = x * 2
    """Function オーソドックス"""
    #f = 3 * 2
    f = theano.function([x], y)
    print f(3)  #6`

    """Function Given込"""
    x = T.dscalar()
    y = T.dscalar()
    c = T.dscalar()

    #ff = (2 * 10.0) * 2 + 5.0 = 45.0
    ff = theano.function([c], x * 2 + y, givens=[(x, c * 10.0), (y, 5.0)])
    print ff(2)  #45.0

    """自動微分"""
    z = (x + 2 * y) ** 2
    gx = T.grad(z, x)  #微分するとdz/dx =2(x+2*y) gxにその式が入る
    fz = theano.function([x, y], z)  #functionを生成
    fgx = theano.function([x, y], gx)  #functionを生成
    print fz(1, 2)
    print fgx(1, 2)


"""
共有変数部分のテスト
Shared
"""


def sharedTest():
    x = T.dscalar('x')
    b = theano.shared(1)
    f = theano.function([x], b * x)
    print f(2)
    b.get_value()
    b.set_value(3)
    print f(2)


"""
Updateの利用方法
これをうまく使うと勾配法を実装できる。
"""


def updateTest():
    """ 簡単なUpdate文の使い方 """
    c = theano.shared(0)
    f = theano.function([], c, updates={c: c + 1})
    print f
    print f()
    print f()
    print f()

    x = T.dvector("x")
    c = theano.shared(0.)
    y = T.sum((x - c) ** 2)
    gc = T.grad(y, c)
    d2 = theano.function([x], y, updates={c: c - 0.05 * gc})

    for i in xrange(3):
        print d2([1, 2, 3, 4, 5])
        print c.get_value()


def testRandomStream():
    """Numpy Random"""
    numpy_rng = numpy.random.RandomState(89677)
    theanoRng = RandomStreams(numpy_rng.randint(2 ** 30))

    #2項分布
    fbino = theanoRng.binomial(size=(1, 3))
    funi = theanoRng.uniform(size=(3, 2))
    #関数化
    f1 = theano.function([], fbino)
    f2 = theano.function([], funi)
    #とりあえず使う
    print f1()
    print f2()


if __name__ == '__main__':
    functionTest()
    sharedTest()
    updateTest()
    testRandomStream()