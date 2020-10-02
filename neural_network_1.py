import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

# 计算损失
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# 一个简单的神经元
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weight) + self.bias
        return sigmoid(total)

# 神经网络
class OurNetWork:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.h1 = Neuron(weight, bias) # 采用单个神经元作为初始化参数
        self.h2 = Neuron(weight, bias)
        self.o1 = Neuron(weight, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

class OurNeuralNetWork:
    def __init__(self):
        # 权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # 截距项
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feed_forward(self, x):
        h1 = sigmoid(self.w1 * x[0] +self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] +self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * x[0] +self.w6 * x[1] + self.b3)
        return o1

    # 训练函数
    def train(self, data, all_y_trues):
        learn_rate = -0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] +self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] +self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * x[0] +self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # 计算偏导数
                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)

                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                # update
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6

                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feed_forward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print('Epoch %d loss:%.3f'%(epoch, loss))



# 初始化
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])
all_y_trues = np.array([1, 0, 0, 1])

# train
network = OurNeuralNetWork()
network.train(data, all_y_trues)

# make some predictions
high = 65    #英寸
weight =133  #磅


zx = np.array([weight - 133, high - 65])
print('zx:%.3f'% network.feed_forward(zx))

if network.feed_forward(zx) >= 0.5:
    print('这是个老男人')
else:
    print('明显是个小姐姐')




