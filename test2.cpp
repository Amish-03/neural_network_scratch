#include <iostream>
#include "nn_utils.h"
#include "activations.h"

int main() {
    // Input vector (3 values)
    float input[3] = {0.5f, -0.2f, 0.1f};

    // Weight matrix (2x3) for 2 neurons taking 3 inputs each
    float weights[6] = {
        0.1f, 0.4f, -0.3f,   // Neuron 1
        0.2f, -0.5f, 0.6f    // Neuron 2
    };

    // Biases for the 2 neurons
    float bias[2] = {0.05f, -0.1f};

    // Output after matrix-vector multiplication
    float linear_output[2];

    // Output after adding bias
    float biased_output[2];

    // Final activated output (after applying sigmoid or ReLU)
    float activated_output[2];

    std::cout << "Input vector: ";
    print_vector(input, 3);

    std::cout << "Weights (flattened 2x3): ";
    print_vector(weights, 6);

    // 1. Linear transformation (matrix-vector product)
    mat_vec_mul(weights, input, linear_output, 2, 3);
    std::cout << "After mat-vec multiplication: ";
    print_vector(linear_output, 2);

    // 2. Add bias
    vec_add(linear_output, bias, biased_output, 2);
    std::cout << "After adding bias: ";
    print_vector(biased_output, 2);

    // 3. Apply activation function (sigmoid here)
    vec_apply_func(biased_output, activated_output, 2, sigmoid);
    std::cout << "After applying sigmoid: ";
    print_vector(activated_output, 2);

    // 4. Try with ReLU for comparison
    vec_apply_func(biased_output, activated_output, 2, relu);
    std::cout << "After applying ReLU: ";
    print_vector(activated_output, 2);

    return 0;
}

