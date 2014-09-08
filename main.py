# -*- coding: utf-8 -*-

__author__ = 'kbysPC'


# ## note function
# function([x],x*2)
# <=同義=>
# y = x*2
# function([x],y)

# ## note fumctionのgiven引数
# >>> x = T.dscalar()
# >>> y = T.dscalar()
# >>> c = T.dscalar()
# >>> ff = theano.function([c], x*2+y, givens=[(x, c*10), (y,5)]) <== xをc*10,yを5で置き換える方法
# >>> ff(2)
# array(45)




# ## note shared
# b = theano.shared(numpy.array([1,2,3,4,5]))
# f = theano.function([x], b * x)
# f(2) ==> array([  2.,   4.,   6.,   8.,  10.])
# b.get_value() 値の参照するkァン数
# b.set_value("値") 値をセットする関数

