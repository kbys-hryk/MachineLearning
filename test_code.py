# coding:utf-8

__author__ = 'kbysPC'

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def test_zeros():
    print np.zeros((3, 4))


def test_arange():
    print np.arange(10)


def test_range():
    for i in range(0, 100, 10):
        print i


def test_function():
    def test():
        return 1, 2, 3

    print test()


def test_argmax():
    x = T.fmatrix('x')
    b = x
    f = T.argmax(b, axis=1)
    ff = theano.function(inputs=[x], outputs=f)

    print ff([[1, 2, 3, 4, 7, 6],
              [1, 2, 3, 4, 7, 9],
              [10, 2, 3, 4, 7, 9]])


def test_softmax():
    # 入力用変数
    x = T.fmatrix('x')
    W = theano.shared(np.zeros((2, 2)), name='W')
    # 重み、バイアスを共有変数で宣言
    b = theano.shared(np.zeros((2,)), name='b')
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

    get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)
    print get_p_y_given_x([[1, 2], [1, 2]])


def test_log():
    x = T.ivector("x")
    y = T.log(x)

    f = theano.function(inputs=[x], outputs=y)
    print f([1, 2, 3, 4, 5])


def test_array():
    x = T.imatrices("x")
    y = T.log(x)[[0, 1], 0]
    f = theano.function(outputs=y, inputs=[x])
    print f([[1, 2, 3, 4, 5],
             [5, 4, 3, 2, 1]
    ])


def test_dot():
    x = T.imatrices("x")
    y = T.dot(x, [[1,2], [1,2], [1,2]])
    f = theano.function(inputs=[x], outputs=y)
    print f([[1, 2, 3], [2, 3, 4]])


def test_sum():
    print np.sum([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], axis=1)


def test_column_stack():
    mean1 = [1, 1]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    N = 12
    x1 = np.random.multivariate_normal(mean1, cov, N / 3)
    label1 = np.zeros((N / 3, 3), dtype=np.int32) + np.array([1, 0, 0])
    print "test_multivariate_normal", x1
    print "test_zeros(N/3,3)", label1

    dataset = np.column_stack((x1, label1))
    print "test_column_stack", dataset


def test_rogistic1():
    K = 10  # 次元数
    a = np.random.rand(K)  # 10個のランダム数生成(ベクトル)

    def softmax(a):
        return np.exp(a) / np.sum(np.exp(a))

    y = softmax(a)


    # Test:要素の合計を出力
    print "random_element:", a
    print "random_element_in_softmax:", y
    print "summation:", np.sum(y)


def test_rogistic2():
    D = 2
    N = 1500
    K = 3

    # データ点を用意
    mean1 = [-2, 2]  # クラス1の平均
    mean2 = [0, 0]  # クラス2の平均
    mean3 = [2, -2]  # クラス3の平均
    cov = [[1.0, 0.8], [0.8, 1.0]]  # 共分散行列（全クラス共通）

    x1 = np.random.multivariate_normal(mean1, cov, N / 3)
    x2 = np.random.multivariate_normal(mean2, cov, N / 3)
    x3 = np.random.multivariate_normal(mean3, cov, N / 3)
    x = np.vstack((x1, x2, x3))

    # 教師ベクトルを用意
    label1 = np.zeros((N / 3, 3), dtype=np.int32) + np.array([1, 0, 0])
    label2 = np.zeros((N / 3, 3), dtype=np.int32) + np.array([0, 1, 0])
    label3 = np.zeros((N / 3, 3), dtype=np.int32) + np.array([0, 0, 1])
    label = np.vstack((label1, label2, label3))

    # 図示
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='g')
    plt.scatter(x3[:, 0], x3[:, 1], c='b')
    plt.show()
    dataset = np.column_stack((x, label))
    np.random.shuffle(dataset)  # データ点の順番をシャッフル

    x = dataset[:, :2]  # 入力
    label = dataset[:, 2:]  # 正解データ

    def softmax(x):
        return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T


    def p_y_given_x(x):
        return softmax(np.dot(x, w.T) + b)

    def grad(x, label):
        error = p_y_given_x(x) - label
        w_grad = np.zeros_like(w)
        b_grad = np.zeros_like(b)

        for j in range(w.shape[0]):
            w_grad[j] = np.mean(error[:, j] * x.T, axis=1)
            b_grad[j] = np.mean(error[:, j])

        return w_grad, b_grad, np.mean(np.abs(error), axis=0)

    w = np.random.rand(K, D)
    b = np.random.rand(K)

    eta = 0.1
    minibatch_size = 500

    # import numpy.linalg as LA

    errors = []
    for _ in range(100):
        for index in range(0, N, minibatch_size):
            _x = x[index: index + minibatch_size]
            _label = label[index: index + minibatch_size]
            w_grad, b_grad, error = grad(_x, _label)
            w -= eta * w_grad
            b -= eta * b_grad

            errors.append(error)
    errors = np.asarray(errors)

    """
    結果の描画
    """
    plt.plot(errors[:, 0])
    plt.plot(errors[:, 1])
    plt.plot(errors[:, 2])
    plt.show()
    bx = np.arange(-10, 10, 0.1)
    by0 = -(w[0, 0] - w[1, 0]) / (w[0, 1] - w[1, 1]) * bx - (b[0] - b[1]) / (w[0, 1] - w[1, 1])
    by1 = -(w[1, 0] - w[2, 0]) / (w[1, 1] - w[2, 1]) * bx - (b[1] - b[2]) / (w[1, 1] - w[2, 1])

    plt.plot(bx, by0)
    plt.plot(bx, by1)

    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='g')
    plt.scatter(x3[:, 0], x3[:, 1], c='b')
    plt.show()

# main処理
if __name__ == "__main__":
    test_dot()