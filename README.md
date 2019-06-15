# Deep Learning Neural Network
A feed forward deep learning neural network that is trained with gradient descent to recognize handwritten digits (training data provided by MNIST)

## Objective
The purpose of this project is to learn the inner workings of modern machine learning and the techniques used. More specifically, my goal is to implement a feed forward neural network that can be used to recognize handwritten digits.

## Introduction
The fundamental principles behind a deep learning neural network is to mimic the mechanisms in a brain, where individual neurons are connected with each other to form a network of neurons. The output of such network can be described with how much each individual neuron becomes excited based on the strength of connections to each stimulus. This behaviour can be modeled mathematically with a discrete map:

[ADD NETWORK ARCHITECTURE ILLUSTRATION HERE]

To calculate how much each neuron is stimulated, each input is simply multiplied with the corresponding 

## Difficulties and Challenges
Despite constantly dealing with matrix multiplication dimension mismatch issues, I met one major obstacle in this project, where I learned how important careful random initialization is. At first, to avoid the problem induced by initializing all thetas to be zero where the backpropagation algorithm makes the same change to every theta, effectively reducing the network architecture to one neuron per layer rendering it essentially useless, I used the default numpy random matrix generator which outputs a matrix of some given size full of random values between 0 and 1 and reshaping them to the appropriate dimensions. However, I did not foresee the huge issue this initialization method would cause to arise. Specifically, when all 
