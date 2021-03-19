import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
    
class Function:
    def __call__(self, input):
        data = input.data
        forward = self.forward(data)
        output = Variable(forward)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
# y = x^2 における、中心近似を求める数値微分
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 合成関数の微分
# y = (e^x*2)^2
def f(x):
    A = Square() # ^x*2
    B = Exp() # e
    C = Square() # (e^x*2)^2

    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)

print(dy)
