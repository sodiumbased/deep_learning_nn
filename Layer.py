from numpy import *

def g(z):
    return 1/(1+pow(e,-z))

class Layer:

    def __init__(self, a=None, theta=None):
        self.theta = theta
        self.a = a

    def activate(self, previous_layer, input_layer=False):
        # inserting the x_0's into the previous layer for theta_0's
        if not input_layer:
            previous_layer.a = insert(previous_layer.a, 0, 1, axis=1)
        self.a = g(matmul(previous_layer.a, previous_layer.theta.transpose()))

    def delt(self, next_layer, m, output_layer=False, next_to_output=False):
        # This function outputs the accumulated gradient for that layer and saves all smaller case delta's as an array of vectors for the previous layers to gradient descent
        self.delta = array([])
        if output_layer:
            for i in range(m):
                self.delta = append(self.delta, self.a[i] - next_layer[i])
            self.delta = reshape(self.delta, (m,len(self.a[0]),1))
        elif next_to_output:
            self.gradiant = zeros(self.theta.shape)
            for i in range(m):
                self.delta = append(self.delta, matmul(self.theta.transpose(), next_layer.delta[i]) * reshape(self.a[i],(self.a[i].size,1)) * (1-reshape(self.a[i],(self.a[i].size,1))))
            self.delta = reshape(self.delta, (m, len(self.a[0]),1))
            for i in range(m):
                self.gradiant += matmul(next_layer.delta[i], reshape(self.a[i], (1,self.a[i].size)))
        else:
            self.gradiant = zeros(self.theta.shape)
            for i in range(m):
                self.delta = append(self.delta, matmul(self.theta.transpose(), next_layer.delta[i][1:]) * reshape(self.a[i],(self.a[i].size,1)) * (1-reshape(self.a[i],(self.a[i].size,1))))
            self.delta = reshape(self.delta, (m, len(self.a[0]),1))
            for i in range(m):
                self.gradiant += matmul(next_layer.delta[i][1:], reshape(self.a[i], (1,self.a[i].size)))