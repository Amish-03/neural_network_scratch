#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// Reads a single 28x28 MNIST image from the given file stream (binary, header skipped).
// Returns true if successful, false otherwise.
// Output: fills the provided float array (size 784) with normalized pixel values [0,1].
bool read_mnist_image(ifstream &file, float* image_out) {
    if (!file.is_open()) return false;

    unsigned char buffer[28 * 28];
    file.read(reinterpret_cast<char*>(buffer), 28 * 28);

    if (file.gcount() != 28 * 28) return false;

    for (int i = 0; i < 28 * 28; i++) {
        image_out[i] = buffer[i] / 255.0f;
    }

    return true;
}

// Call this once after opening the file to skip the 16-byte header.
inline void skip_mnist_header(ifstream &file) {
    file.ignore(16);
}

#endif // MNIST_LOADER_H
