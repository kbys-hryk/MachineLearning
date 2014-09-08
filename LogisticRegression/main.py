__author__ = 'kbysPC'
# coding:utf-8
import numpy
import theano
import theano.tensor as T

# 入力用変数
x = T.fmatrix('x')
y = T.lvector('y')

# 重み、バイアスを共有変数で宣言
b = theano.shared(numpy.zeros((10,)), name='b')
W = theano.shared(numpy.zeros((784, 10)), name='W')

# ソフトマックス関数に
# (x_i * w_i)+b_i
# 入力を入れる数式
p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)
get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)

# 確率の最大値を取ってくる関数,評価
y_pred = T.argmax(p_y_given_x, axis=1)
classify = theano.function(inputs=[x], outputs=y_pred)

# 損失関数
loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
"""
mean:平均を求める
arange:連番生成
 (ex)
    >>>T.arange(3.0)
    array([0. , 1. , 2.])
y.shape[0]:y軸の縦要素の数
"""

"""
Pythonのクラス定義
 class Class名(継承するクラス):
    def _init_(self,[コンストラクタの引数])

"""


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # 重み行列を0で初期化,行:n_out 列:n_in
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX), name='W')
        # initialize the biases b as a vector of n_out 0s
        # バイアス行列を0で初期化, 要素数:n_out
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX), name='b')

        # compute vector of class-membership probabilities in symbolic form
        # それぞれの入力に対し重みをかけてバイアス項を足す
        f = T.dot(input, self.W) + self.b
        # その結果をソフトマックス関数に入れる
        self.p_y_given_x = T.nnet.softmax(f)


        # compute prediction as class whose probability is maximal in
        # symbolic form
        # 行列の各列の最大値のインデックスを返す関数 argmax
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    # 損失関数の値を返す関数
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

print "test"