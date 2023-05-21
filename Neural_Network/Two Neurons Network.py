import numpy as np

# Acitvation Function
def sigmoid(x):
    return ( np.exp(x) / ( 1 + np.exp(x) ) )

# Derivative of the activtaion function
def deriv_sigmoid(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

# Loss in predection
def mse_loss(true, pred):
    return ((true - pred) ** 2).mean()


class NeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.bias1 = np.random.normal()
        self.bias2 = np.random.normal()
        self.bias3 = np.random.normal()



    # def hiddenLayer(self, x):
    #     total = np.dot(self.weights, x) + bias
    #     return sigmoid(total)
    
    def feedForward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.bias1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.bias2)

        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.bias3)

        return (o1)
    

    def train(self, data, all_y_true):

        # Learn Rate
        learn_rate = 0.1
        epochs = 10000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):


                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.bias1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.bias2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.bias3
                o1 = sigmoid(sum_o1);
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron O1
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_ypred_d_bias3 = deriv_sigmoid(sum_o1)
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)


                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_bias1 = deriv_sigmoid(sum_h1)


                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_bias2 = deriv_sigmoid(sum_h2)


                #  Updating value

                # Biases and Weights 

                #  Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.bias1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_bias1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.bias2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_bias2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.bias3 -= learn_rate * d_L_d_ypred * d_ypred_d_bias3


            if(epoch % 10 == 0):
                y_preds = np.apply_along_axis(self.feedForward, 1, data)
                loss = mse_loss(all_y_true, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))



# Data set

data = np.array([
    [-2, -1],  # Alice
    [25, 6],    # Bob
    [17, 4],     # Charlie
    [-15, -6]     # Diana
])

all_y_true = np.array([
    1, 0, 0, 1
])

network = NeuralNetwork()
network.train(data, all_y_true)

#  Testing
emily = np.array([-7, -3])
clark = np.array([20, 2])

value = ['Male', 'Female']

print('Emily: %s' % value[round(network.feedForward(emily))])
print('Clark: %s' % value[round(network.feedForward(emily))])