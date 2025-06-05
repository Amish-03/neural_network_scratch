#include "header1.h"
#include <fstream>

int main() {
    ifstream file("t10k-images.idx3-ubyte", ios::binary);
    if (!file.is_open()) {
        cout << "Failed to open file\n";
        return 1;
    }

    skip_mnist_header(file);

    float image[28 * 28];
    if (read_mnist_image(file, image)) {
        // Print the image matrix
        for (int i = 0; i < 28 * 28; i++) {
            cout << image[i] << " ";
            if ((i + 1) % 28 == 0) cout << "\n";
        }
    } else {
        cout << "Failed to read image\n";
    }

    file.close();
    return 0;
}
