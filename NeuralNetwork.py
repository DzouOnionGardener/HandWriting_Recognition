import numpy as np
import scipy
class neuralNetwork:
    def __init__(self, inNodes, hiddenNodes, outNodes, learningRate):
        self.in_nodes = inNodes
        self.hidden_nodes = hiddenNodes
        self.out_nodes = outNodes
        self.LR = learningRate

        self.inputWeights = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.in_nodes))
        self.outputWeights = np.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hidden_nodes))
        self.activation = lambda x: scipy.special.expit(x)

    def training(self, inputList, targetList):
        inputs = np.array(inputList, ndmin=2).T
        target = np.array(targetList, ndmin=2).T

        hidden_inputs = np.dot(self.inputWeights, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        new_inputs = np.dot(self.outputWeights, hidden_outputs)
        new_outputs = self.activation(new_inputs)

        error = target - new_outputs
        self.outputWeights += self.LR * np.dot((error * new_outputs * (1.00 - new_outputs)), np.transpose(hidden_outputs))
        self.inputWeights += self.LR * np.dot((error*hidden_outputs * (1.00 - hidden_outputs)), np.transpose(inputs))

        pass


    def search(self, inputList):
        inputs = np.array(inputList, ndmin=2).T
        hidden_inputs = np.dot(self.inputWeights, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        new_inputs = np.dot(self.outputWeights, hidden_outputs)
        new_outputs = self.activation(new_inputs)

        return new_outputs