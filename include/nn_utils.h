#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <iostream>
#include <math.h>

// -------------------- Vector & Matrix Utilities --------------------

// Dot product of two vectors of length n

struct DenseLayer {
    int input_size;
    int output_size;
    float* weights;  // [output_size x input_size]
    float* biases;   // [output_size]
    float* output;   // [output_size]
};

float randf();

void forward_dense(DenseLayer &layer, float* input, float (*activation)(float));


void init_dense_layer(DenseLayer &layer, int input_size, int output_size);

inline float dot_product(const float* vec1, const float* vec2, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Matrix-vector multiplication: matrix is rows x cols, vec is cols x 1
// Output vector must be preallocated with size = rows
inline void mat_vec_mul(const float* matrix, const float* vec, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i] += matrix[i * cols + j] * vec[j];
        }
    }
}

// Vector addition: output[i] = vec1[i] + vec2[i]
inline void vec_add(const float* vec1, const float* vec2, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = vec1[i] + vec2[i];
    }
}

// Apply function element-wise: output[i] = func(input[i])
inline void vec_apply_func(const float* input, float* output, int n, float (*func)(float)) {
    for (int i = 0; i < n; i++) {
        output[i] = func(input[i]);
    }
}

// Print a vector (for debugging)
inline void print_vector(const float* vec, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

#endif // NN_UTILS_H

