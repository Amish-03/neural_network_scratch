#include <iostream>
#include "include/nn_utils.h"
#include "include/activations.h"

// Identity activation for testing (no change)
float identity(float x) {
    return x;
}

int main() {
    const int input_size = 5;
    const int output_size = 5;

    // Initialize dense layer
    DenseLayer layer;
    init_dense_layer(layer, input_size, output_size);

    // Create dummy input vector with some values
    float input[input_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Forward pass with identity activation (just weighted sum + bias)
    forward_dense(layer, input, identity);

    std::cout << "Layer output before softmax:\n";
    for (int i = 0; i < output_size; ++i) {
        std::cout << layer.output[i] << " ";
    }
    std::cout << "\n\n";

    // Apply softmax on the output array (in-place)
    softmax(layer.output, output_size);

    std::cout << "Layer output after softmax:\n";
    for (int i = 0; i < output_size; ++i) {
        std::cout << layer.output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup dynamically allocated memory
    delete[] layer.weights;
    delete[] layer.biases;
    delete[] layer.output;

    return 0;
}
