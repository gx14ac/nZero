import numpy as np

class Variable:
    def __init__(self, data):
        self.nd_array_data = data
        self.nd_array_grad = None

class Function:
    def __call__(self, variable):
        x = variable.nd_array_data
        y = self.forward(x)
        output = Variable(y)
        self.input = variable # for backpropagation
        return output
    
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
        x = self.input.nd_array_data # forwardの値
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
y.nd_array_grad = np.array(1.0) # 逆伝播の時はdx/dy=1.0
b.nd_array_grad = C.backward(y.nd_array_grad)
a.nd_array_grad = B.backward(b.nd_array_grad)
x.nd_array_grad = A.backward(a.nd_array_grad)
print(x.nd_array_grad)