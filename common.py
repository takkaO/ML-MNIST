import numpy as np

class ReLU:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)	# 0以下がTrueなマスクを作成
		out = x.copy()
		out[self.mask] = 0	# Trueなところに0を代入
		return out
	
	def backward(self, dout):
		dout[self.mask] = 0
		return dout


class Affine:
	def __init__(self, W, B):
		self.W = W
		self.X = None
		self.B = B
		self.dw = None
		self.db = None
	
	def forward(self, x):
		self.X = x
		return np.dot(self.X, self.W) + self.B
	
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dw = np.dot(self.X.T, dout)
		self.db = np.sum(dout, axis=0)

		return dx

class SoftmaxWithLoss:
	def __init__(self):
		self.y = None
		self.loss = None
		self.t = None
	
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = crossEntropy(self.y, self.t)

		return self.loss
	
	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size

		return dx


def crossEntropy(result, teacher):
	delta = 1e-7  # logが発散しないために微小値を入れておく
	if result.ndim == 1:
		result = result.reshape(1, result.size)
		teacher = teacher.reshape(1, teacher.size)
	return -np.sum(teacher * np.log(result + delta)) / result.shape[0]


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def numerical_gradient(func, x):
	h = 1e-4
	grad = np.zeros_like(x)

	# ループ用にイテレータを作成
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = func(x)  # f(x+h)

		x[idx] = tmp_val - h
		fxh2 = func(x)  # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)

		x[idx] = tmp_val  # 値を元に戻す
		it.iternext()
	return grad
