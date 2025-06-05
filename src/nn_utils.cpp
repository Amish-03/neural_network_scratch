#include <iostream>
#include <cstdlib>  // For rand()
#include <cmath>    // For math functions like pow()
#include "../include/nn_utils.h"  // Your header with declarations
#include "../include/activations.h"
// Helper: Generate random float in range [-1, 1]
float randf() {
    return 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
}

// Initialize DenseLayer with random weights and zero biases
void init_dense_layer(DenseLayer &layer, int input_size, int output_size) {
    layer.input_size = input_size;
    layer.output_size = output_size;

    // Allocate memory
    layer.weights = new float[output_size * input_size];
    layer.biases = new float[output_size];
    layer.output = new float[output_size];

    // Initialize weights with small random values
    for (int i = 0; i < output_size * input_size; ++i) {
        layer.weights[i] = randf() * 0.01f;  // Small random weights
    }

    // Initialize biases to zero
    for (int i = 0; i < output_size; ++i) {
        layer.biases[i] = 0.0f;
        layer.output[i] = 0.0f;  // initialize output to zero
    }
}

// Forward pass through dense layer: output = activation(W * input + b)


void forward_dense(DenseLayer &layer, float* input, float (*activation)(float)) {
    for (int i = 0; i < layer.output_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < layer.input_size; ++j) {
            sum += layer.weights[i * layer.input_size + j] * input[j];
        }
        sum += layer.biases[i];
        layer.output[i] = activation(sum);
    }
}



