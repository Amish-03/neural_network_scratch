# Neural Network from Scratch in C++

This project is an attempt to implement a basic neural network from scratch in C++.

## Features
- One hidden layer neural network
- Uses ReLU and Softmax activation functions
- Forward pass implemented
- Structure designed for MNIST input (28x28 = 784 features)

## Files
- `include/nn_utils.h`: Contains structure definitions and function declarations
- `include/activations.h`: Contains activation function declarations
- `src/nn_utils.cpp`: Function definitions for layer initialization and forward pass
- `test/test_main.cpp`: Sample test for the network

## Current Progress
- Defined DenseLayer structure
- Implemented ReLU and Softmax activation functions
- Forward pass for a dense layer is working
- Output tested with dummy inputs

## Next Steps
- Add more layers
- Implement training (backpropagation)
- Load and test on real MNIST data

## Build
Use g++ or any C++ compiler to build.
Make sure to include the `include/` directory.


