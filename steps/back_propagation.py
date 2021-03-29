import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.nd_array_data = data
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
            x, y = f.input, f.output
            x.nd_array_grad = f.backward(y.nd_array_grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input_variable):
        x = input_variable.nd_array_data
        y = self.forward(x)
        variable = Variable(as_array(y))
        variable.set_creator(self)   # 出力変数に関数を覚えさせる
        self.input = input_variable  # for backpropagation
        self.output = variable       # 出力も覚えておく
        return variable

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
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
    def backward(self, gy):
        x = self.input.nd_array_data  # forwardの値
        gx = 2 * x * gy
        return gx

# ネイピア指数関数
# 1>a>0


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.nd_array_data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


# 中心差分近似で微分を求める
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.nd_array_data - eps)
    x1 = Variable(x.nd_array_data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.nd_array_data - y0.nd_array_data) / (2 * eps)


x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.nd_array_grad)


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
