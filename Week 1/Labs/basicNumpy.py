import numpy as np
import matplotlib
import math

# The following is the sigmoid function

def sigmoid(x):
    return (1 / (1 + np.exp(-1 * x)))

# Example of a vector operation
x = np.array([1, 2, 3])
print(x)

print(sigmoid(x))

# Derivative of Sigmoid function
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

# Changing image data into a (length * width * dimmensions, 1) vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    image = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)
    return image

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image))) 

# Function to normalize rows to make gradient run faster
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x / x_norm
    return x_normalized

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


# Softmax function to normalize vectors in a different way
def softmax(x):
    # Applies element-wise exponent
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum

    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


def L1(yhat, y):
    loss = np.sum(abs(yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(yhat, y):
    diffMatrix = yhat-y
    loss = np.dot(diffMatrix, diffMatrix)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))