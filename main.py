import numpy as np
from collections import OrderedDict
from common import *


class TwoLayer:
	def __init__(self):
		input_num = 784
		hidden_num = 100
		output_num = 10

		self.params = {}
		self.params['W1'] = 0.01 * np.random.randn(input_num, hidden_num)
		self.params['b1'] = np.zeros(hidden_num)
		self.params['W2'] = 0.01 * np.random.randn(hidden_num, hidden_num)
		self.params['b2'] = np.zeros(hidden_num)
		self.params['W3'] = 0.01 * np.random.randn(hidden_num, output_num)
		self.params['b3'] = np.zeros(output_num)

		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['ReLU'] = ReLU()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['ReLU'] = ReLU()
		self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

		self.lastLayer = SoftmaxWithLoss()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def calcGrads(self, x, t):
		# forward
		self.loss(x, t)

		#backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads = {}
		grads['W1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
		grads['W2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db
		grads['W3'], grads['b3'] = self.layers['Affine3'].dw, self.layers['Affine3'].db
		return grads

	def train(self, x, t, iters):
		lr = 0.01  # 学習率
		batch_size = 100
		for i in range(iters):
			batch_mask = np.random.choice(x.shape[0], batch_size)
			x_batch = x[batch_mask]
			t_batch = t[batch_mask]
			grad = self.calcGrads(x_batch, t_batch)
			for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
				self.params[key] -= lr * grad[key]
			if i % 100 == 0:
				print("Loss: ", self.loss(x_batch, t_batch))


def load_img(fpath):
	with open(fpath, "rb") as f:
		# バッファからnumpy行列を作成する
		# 16進数データで見たほうがわかりやすい
		# 先頭16バイトは説明データなので省略
		data = np.frombuffer(f.read(), np.uint8, offset=16)
	# 28x28=784
	data = data.reshape(-1, 784)
	return data


def to_one_hot(label):
    T = np.zeros((label.size, 10))
    for i in range(label.size):
        T[i][label[i]] = 1
    return T


def load_label(fpath, onehot=True):
	with open(fpath, "rb") as f:
		# バッファからnumpy行列を作成する
		# 16進数データで見たほうがわかりやすい
		# 先頭8バイトは説明データなので省略
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	if onehot:
		labels = to_one_hot(labels)
	return labels


def main():
	print("Train start!")
	tl = TwoLayer()

	x = load_img("train-images.idx3-ubyte")
	all_t = load_label("train-labels.idx1-ubyte", False)
	t = to_one_hot(all_t)

	x_test = x[0:10]
	x_train = x[10:]
	t_test = t[0:10]
	t_train = t[10:]

	tl.train(x_train, t_train, 10000)
	print("Train finish\n")
	
	print("p  t\n")
	for i in range(10):
		print(np.argmax(tl.predict(x[i])), end="  ")
		print(all_t[i])

if __name__ == "__main__":
	main()
