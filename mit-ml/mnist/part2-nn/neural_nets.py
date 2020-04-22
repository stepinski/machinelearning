import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return x*(x > 0)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    ret = 1 if x>0 else 0
    return ret

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 1
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        # self.training_points=[((-7, -9), -16), ((7, 7), 14), ((-2, -5), -7), ((0, 5), 5), ((3, 8), 11), ((-9, 2), -7), ((-4, -7), -11), ((9, -6), 3), ((6, 2), 8), ((-5, 7), 2)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
 

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1
        x=input_values
        w1=self.input_to_hidden_weights
        w2=self.hidden_to_output_weights
        b=self.biases
        f=output_layer_activation
        z1=w1*x+b
        tmp=np.vectorize(rectified_linear_unit)
        a1=tmp(z1)
        z2=np.multiply(a1.T,w2)
        o=f(z2)
        fprim=np.vectorize(output_layer_activation_derivative)
        E0=np.multiply((y-o),fprim(z2))
        tmprim=np.vectorize(rectified_linear_unit_derivative)
        Eh=np.multiply(np.multiply(w2,E0),tmprim(z1).T)
        deltab=Eh
        deltaw2=np.multiply(E0,a1.T)
        deltaw1=np.multiply(Eh,x).sum(axis=0)

        bnew=b-np.multiply(self.learning_rate,deltab.T)
        w2new=w2-self.learning_rate*deltaw2
        w1new=w1-self.learning_rate*deltaw1.T
        self.biases = bnew
        self.input_to_hidden_weights = w1new
        self.hidden_to_output_weights=w2new


        # Calculate the input and activation of the hidden layer

# t=1
# x=3
# w1=0.01
# w2=-5
# b=-1
# z1=w1*x
# a1=ReLU(z1)
# z2=w2*a1+b
# y=sigmoid(z2)
# c=0.5*(float(y)-t)*(float(y)-t)  
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights,input_values)+ self.biases
        # print("hidden_weighted:%s"%str(hidden_layer_weighted_input.shape))
        # (3 by 1 matrix)
        gx=np.vectorize(rectified_linear_unit)
        hidden_layer_activation = gx(hidden_layer_weighted_input)
        # print("hidden_active:%s"%str(hidden_layer_activation.shape))
        

        output =   np.dot(self.hidden_to_output_weights,hidden_layer_activation)
        activated_output = output_layer_activation(output)
        # print("active_out:%s"%str(activated_output.shape))
        ### Backpropagation ###

        # Compute gradients
        # lossy=1/2*(activated_output-y).T*(activated_output-y)
        dLoss_Yh=y-activated_output
        g = np.vectorize(output_layer_activation_derivative)

# ========================================
        dLoss_Z2 = dLoss_Yh * g(activated_output)  
        dLoss_A1 = np.dot(self.hidden_to_output_weights.T,dLoss_Z2)
        dLoss_W2 = 1./activated_output.shape[1] * np.dot(dLoss_Z2, activated_output.T)
        dLoss_b2 = 1./ activated_output.shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 

        g2 = np.vectorize(rectified_linear_unit_derivative)                    
        dLoss_Z1 = np.dot(dLoss_A1.T ,g2(hidden_layer_weighted_input))
        # dLoss_A0 = np.dot(self.input_to_hidden_weights.T,dLoss_Z1)
        dLoss_W1 = 1./input_values.shape[1] * np.dot(dLoss_Z1,input_values.T)
        dLoss_b1 = 1./input_values.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1])) 
        # print(str(dLoss_W1))
        # print(str(dLoss_b1))
        # print("okejko")

# ===============================================

        # dy_z2=g(activated_output)
        # print("dlossy:%s"%str(dlossy.shape))

        # dlossZ2=np.dot(dlossy,dy_z2)
        # print("dlossZ2:%s"%str(dlossZ2.shape))
        # dlossA1=np.dot(self.hidden_to_output_weights.T,dlossZ2)
        # dlossW2=1./hidden_layer_activation.shape[1]*np.dot(dlossZ2,hidden_layer_activation.T)
        # print("dlossW2:%s"%str(dlossW2.shape))

        # g2 = np.vectorize(rectified_linear_unit_derivative)
        # da1_z1=g2(hidden_layer_weighted_input)
        # dlossZ1=np.dot(dlossA1,da1_z1)
        
        # print("hidden_layer_weighted_input:%s"%str(hidden_layer_weighted_input.shape))
        # print("da1_z1:%s"%str(da1_z1.shape))
        # print("dlossA1:%s"%str(dlossA1.shape))

        # tstsas=1./input_values.shape[1] * np.dot(dlossZ1,input_values.T)
        # print("tstsas:%s"%str(tstsas.shape))
        # print("dlossZ1:%s"%str(dlossZ1.shape))
        # dlossA0=np.dot(self.input_to_hidden_weights.T,dlossZ1)
        # dz1_w1=input_values
      
        # dlossW1=np.dot(tstsas,input_values.T)
        # dlossB1=np.dot(tstsas,np.ones([dlossZ1.shape[1],1]))
 

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases-self.learning_rate*dLoss_b1
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*dLoss_W1
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*dLoss_W2
        # print("passx")
        

    # def predict(self, x1, x2):

    #     input_values = np.matrix([[x1],[x2]])

    #     # Compute output for a single input(should be same as the forward propagation in training)
    #     hidden_layer_weighted_input = # TODO
    #     hidden_layer_activation = # TODO
    #     output = # TODO
    #     activated_output = # TODO

    #     return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)
            print("bias")
            print(self.biases)
            print("w2")
            print(self.hidden_to_output_weights)
            print("w1")
            print(self.input_to_hidden_weights)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
# x.test_neural_network()



