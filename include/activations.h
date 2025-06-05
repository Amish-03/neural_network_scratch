#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>

// Sigmoid activation
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Sigmoid derivative
inline float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// ReLU activation
inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ReLU derivative
inline float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

inline void softmax(float* input, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        input[i] = std::exp(input[i] - max_val);  // for numerical stability
        sum += input[i];
    }

    for (int i = 0; i < length; ++i) {
        input[i] /= sum;
    }
}


#endif // ACTIVATIONS_H

