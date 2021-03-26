import numpy as np


class Variable:
    def __init__(self, data):
        self.nd_array_data = data
        self.nd_array_grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. 関数を取得
        if f is not None:
            x = f.input  # 2. 関数の入力を取得
            # 3. 関数のbackward(forward値の微分を返す)
            x.nd_array_grad = f.backward(self.nd_array_grad)
            # 4. 自分より一つ前の変数のackwardメソッドを呼ぶ
            x.backward()



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


# 順伝播
# 0.5の時の数値微分を求める
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 逆伝播
y.nd_array_grad = np.array(1.0)
y.backward()
print(x.nd_array_grad)
