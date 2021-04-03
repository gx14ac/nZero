import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        # nd_array_data = 入力変数
        self.nd_array_data = data
        # nd_array_grad = ある関数に対しての偏微分値(ある時点での関数の傾き具合)
        self.nd_array_grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 逆伝播ロジック(動的配列生成)
    # 1. 関数取得
    # 2. 関数の入出力を取得
    # 3. 関数のbackwardメソッドを呼ぶ
    # 4. 一つ前の関数をリストに追加
    def backward(self):
        if self.nd_array_grad is None:
            self.nd_array_grad = np.ones_like(self.nd_array_data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # 出力変数をリストに
            gys = [output.nd_array_grad for output in f.outputs]
            # 出力変数に対して、逆伝播を行う(偏微分)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 前関数の引数と逆伝播された値
            for x, gx in zip(f.inputs, gxs):
                x.nd_array_grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *input_variables):
        xs = [x.nd_array_data for x in input_variables]
        ys = self.forward(*xs)  # アンパッキング、リストの要素を展開して渡す
        # タプルでない場合の対応
        if not isinstance(ys, tuple):
            ys = (ys,)
        variables = [Variable(as_array(y)) for y in ys]

        for variable in variables:
            variable.set_creator(self)   # 出力変数に関数を覚えさせる
        self.inputs = input_variables  # for backpropagation
        self.outputs = variables       # 出力も覚えておく
        return variables if len(variables) > 1 else variables[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    # 伝播
    # y = x^2
    def forward(self, x):
        y = x ** 2
        return y

    # 逆伝播
    # gy = ndarrayインスタンスの微分
    # y = x^2の微分が、dy/dx = 2x
    def backward(self, gys):
        x = self.inputs[0].nd_array_data  # forwardの値
        gx = 2 * x * gys
        return gx


# 足し算の伝播と逆伝播
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

# ネイピア指数関数
# 1>a>0


class Exp(Function):
    def forward(self, xs):
        y = np.exp(xs)
        return y

    def backward(self, gys):
        x = self.input.nd_array_data
        gx = np.exp(x) * gys
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


# 中心差分近似で微分を求める
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.nd_array_data - eps)
    x1 = Variable(x.nd_array_data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.nd_array_data - y0.nd_array_data) / (2 * eps)


# x = Variable(np.array(0.5))
# y = square(exp(square(x)))
# y.backward()
# print(x.nd_array_grad)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.nd_array_data)
print(x.nd_array_grad)
print(y.nd_array_grad)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.nd_array_data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        excepted = np.array(6.0)
        self.assertEqual(x.nd_array_grad, excepted)

    def test_grad_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flag = np.allclose(x.nd_array_grad, num_grad)
        self.assertTrue(flag)
