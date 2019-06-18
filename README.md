# Deep Learning Neural Network

A feed forward deep learning neural network that is trained with gradient descent to recognize handwritten digits (training data provided by MNIST) written in python (numpy included)

## Objective

The purpose of this project is to solve the classic problem of machines not being able to read human handwritten digits with a modern neural network, where techniques like gradient descent are used.

## Usage

Train the neural network further:
```
$ python3 train.py
```
Test the neural network's output against the testing dataset:
```
$ python3 test.py
```
Visualize one of the training images:
```
$ python3 show_training_images.py
```
Try your own handwritten digit and see the network prediction:
```
$ python3 paint.py
```
(Press the green button to submit and red button to clear; if a matplotlib graph pops up close it then view the result in the parent console)

## Introduction

The fundamental principle behind a deep learning neural network is to mimic the mechanisms in a brain, where individual neurons are connected with each other to form a network of neurons. The output of such network can be described with how much each individual neuron becomes excited, or activated, based on the strength of connections to each stimulus. This behaviour can be modeled mathematically with a discrete map:

[ADD NETWORK ARCHITECTURE ILLUSTRATION HERE] cap: mathematical equations explained later

To calculate how much each neuron is stimulated, each input is simply multiplied with the corresponding weight, quantifying how sensitive the neuron's activation is to that input, then fed into an activation function. Naturally, it is quite easy to traverse forwards in this network and fetch an accurate prediction from the output of the network, provided that the right parameters are given. This raises the question: how does one find these parameters?

## The Problem

The goal of the neural network is to read a 28x28 image and identify which digit it is from 0 to 9. This is what's known as a classification problem, where the answer lies in a certain discrete category. This neural network uses the "one versus all" method, where each output neuron signifies the probability of the image representing one digit compared to all other categories, denoted as *P(h<sub>θ</sub>(x)=1|x;θ)* (where 1 is true and 0 is false). For example, *P(h<sub>θ</sub>(x)<sub>j</sub>=1|x;θ) = 0.86* means that there is a 86% chance that this picture represents the digit *j* and a 14% that it's not.

Luckily, there is a free database called the MNIST dataset, which provided me with 60,000 sets of training data (images with their respective labels) and an additional 10,000 testing sets.

To satisfy these requirements, I set up my neural network architecture this way: An input layer consists of 785 neurons (the first one being 1 which I will explain bellow), two hidden layers with 50 neurons each (chosen arbitrarily), and an output layer that has 10 neurons (applying the "one versus all" strategy to every digit).

## Usage of Linear Algebra

While it is possible to simulate the network by actually creating a network of neuron objects, it is computationally beneficial to use linear algebra, especially matrix multiplication where applicable, to simplify/accelerate the calculations. For example, when calculating the activation of the first hidden layer, one has the choice to either use three nested for loops to calculate the sum of products of each input neuron and their respective parameters over every training data set, or to simply multiply all the inputs as one matrix and all the parameters as another and end up with an answer matrix.

## The Hypothesis, Activation Function and Forward Propagation

!["all details worked out on a blackboard"](https://github.com/sodiumbased/deep_learning_nn/blob/master/report_images/work.png)
Fig. 2: details of the network written down on a blackboard

which makes the "activation" of that layer a 60000x785 matrix, denoted as *a<sup>(1)</sup> ϵ ℝ<sup>60000x785</sup>*, and its parameters, which are used to calculate the activation of the next layer, denoted as *θ*, *θ<sup>(1)</sup> ϵ ℝ<sup>50x785</sup>*

## The Cost Function

Since the output of each neuron in the neural network *a<sub>j</sub><sup>(4)</sup> ϵ* [0, 1] , it is much better to use a logarithmic cost function than simply taking the squared difference between the network output and the training label. More specifically, when *y<sub>j</sub> = 1* for some j ϵ [1,m] given m training sets (or images in this case), the cost function is *-ln h<sub>θ</sub>(x)* where if h<sub>θ</sub>(x) approaches 0, which is where we don't want it to be, cost will approach infinity. Similarly, when *y<sub>j</sub> = 0*, the cost is modeled as -ln (1 - h<sub>θ</sub>(x)), where as h<sub>θ</sub>(x) approaches 1, cost approaches infinity. Furthermore, in order to prevent overfitting, a phenomenon that occurs when the parameters in the neural work are too large, resulting in some complex but unnecessary decision boundaries (the curves describing how the network makes decisions i.e. in "one versus all" classification scenarios) that only fits the training data well but fails to extrapolate when processing examples outside of the training set. As a result, the average size of each parameter must also be taken into consideration into the cost of the network, multiplied by some constant *λ ϵ ℝ*. This practice is known as regularization. Here is the compounded cost function:

[insert cost function] caption: the cost function quantifies the average loss in training when compared to the actual training labels

## Backpropagation

This is the hard part of this project where we find the right thetas such that the output of the network matches the training labels as close as possible. Here I used a optimization technique called gradient descent. This technique is very intuitive because the thetas are updated according to their impact to the output of the cost function. Concretely, by subtracting the partial derivative of the cost function with respect to one theta means that: if the derivative is positive, the theta is reduced to lower the cost, and vice versa when the derivative is negative. This derivative is often multiplied by a constant called learning rate 

It is worth mentioning that if the regularization constant λ is too small, it would have little effect in preventing overfitting; however too large a λ can result in the network underfitting even the training data which is not ideal either.

## Difficulties and Challenges

Despite constantly dealing with matrix multiplication dimension mismatch issues, I met one major obstacle in this project, where I learned how important careful random initialization is. At first, to avoid the problem induced by initializing all thetas to be zero where the backpropagation algorithm makes the same change to every theta, effectively reducing the network architecture to one neuron per layer rendering it essentially useless, I used the default numpy random matrix generator which outputs a matrix of some given size full of random values between 0 and 1 and reshaping them to the appropriate dimensions. However, I did not foresee the huge issue this initialization method would cause to arise. Specifically, when all 

## Conclusion
