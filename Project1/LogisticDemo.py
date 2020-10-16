#!/usr/bin/python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/11 14:15
# @Author  : Yu LiXinQian
# @Email : 1316087165@qq.com
# @File : LogisticDemo.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# load_dataset is a function which reads the datasets from flie "datassets".
# The function return a tuple containing (train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes)

'''
train_set_x_orig: Contain the image dataset from training set that each image is of shape (num_px, num_px, 3) where 3 is 
 for the 3 channels (RGB).
train_set_y_orig: Contain the label of each image in the training set. Cat (y=1) or non-cat (y=0)
test_set_x_orig: Contain the image dataset for testing.
test_set_y_orig: Contain the label of each image in test dataset.
classes: Saves two strings of data saved in bytes type: [b'non-cat' b'cat']。
'''

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print("-"*15+"Display The Data"+"-"*15)
print("The number of training examples: {}".format(train_set_x_orig.shape[0]))
print("The number of testing examples: {}".format(test_set_x_orig.shape[0]))
print("Height/Width of each image: {0} X {0}".format(train_set_x_orig.shape[1]))
print("Size of each image(RGB): {}".format(test_set_x_orig.shape[1:]))
print("Train_set_x shape: {}".format(train_set_x_orig.shape))
print("Train_set_y shape: {}".format(train_set_y.shape))
print("Test_set_x shape: {}".format(test_set_x_orig.shape))
print("Test_set_y shape: {}".format(test_set_y.shape))

# Flatten each RGB image to transform the the dimension from (209, 64, 64, 3) to (209,12288).
# The first dimension refers to the number of training examples ,the second dimension refers to the every image.

train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

print("-"*15+"After Flattened"+"-"*15)
print("Train_set_x Shape: {}".format(train_set_x.shape))
print("Test_set_x Shape: {}".format(test_set_x.shape))

'''
One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract 
the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole
numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every 
row of the dataset by 255 (the maximum value of a pixel channel).
'''

train_set_x = train_set_x/255
test_set_x = test_set_x/255

def sigmoid(z):
    """
        Compute the sigmoid of z

        Arguments:
        x -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
    """
    sig=1/(1+np.exp(-z))
    return sig

# Implement parameter initialization in the cell below
def initialize_with_zeros(dim):
    """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
    """
    w=np.zeros((dim,1))
    b=0

    return w,b

# Implement a function propagate() that computes the cost function and its gradient.
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    # forward
    m=X.shape[1]
    z=np.dot(w.T, X)+b
    activation = sigmoid(z)
    cost = (- 1 / m) * np.sum(Y * np.log(activation) + (1 - Y) * (np.log(1 - activation)))
    # backward
    dw = (1 / m) * np.dot(X, (activation - Y).T)
    db = (1 / m) * np.sum(activation - Y)
    grads = {"dw": dw,"db": db}
    return grads,cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,"b": b}

    grads = {"dw": dw,"db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d= model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

print ('\n' + "-------------------------------------------------------" + '\n')
learning_rates = [0.01, 0.005, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()