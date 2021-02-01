import numpy
import matplotlib
import math

# The following is the sigmoid function

def basic_sigmoid(x):
    return (1 / (1 + math.exp(-1 * x)))


