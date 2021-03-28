import numpy as np


class Variable:
    def __init__(self, data):
        self.nd_array_data = data
        self.nd_array_grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.nd_array_grad = f.backward(y.nd_array_grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input_variable):
        x = input_variable.nd_array_data
        y = self.forward(x)
        variable = Variable(y)
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
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.nd_array_grad = np.array(1.0)
y.backward()
print(x.nd_array_grad)
