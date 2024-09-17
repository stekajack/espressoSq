
# EspressoSq

EspressoSq is a high-performance library for structure factor calculations, implemented in C++ with SIMD optimizations and physics-based improvements. It includes a setup script to generate a Python wrapper using Cython, making it accessible for both C++ and Python users.

## Features

- **Fast Calculations**: Optimized with SIMD instructions for superior speed.
- **Physics-Based Optimizations**: Ensures accurate and efficient computations, where you can choose how many wavevectors to use for sampling and how many qs you want to calculate (will choose q so that they are uniformly distributed on a log scale)
- **Cross-Platform**: Primarily implemented in C++, with a Python wrapper for ease of use in Python environments.
- **Minimal Dependencies**: I suggest GCC for compilation, but otherwise dependency-free.

## Installation

### Prerequisites

- **GCC Compiler**: Ensure you have GCC installed on your system.
- **Python with Cython**: To use the Python wrapper, you need Python installed.

### Steps

1. Clone the repository:
2. Navigate to the directory:
    mkdir build
    cd build
    cmake ..
    make
3. (Optional) Run the setup script:
    python3 setup.py build_ext --inplace

## Usage

### In C++

Include the header file and link against the compiled library in your C++ project:
```cpp
#include "sq_avx.hpp"

// Example usage
int main() {
    // Define the parameters for the test
    const unsigned int num_particles = 9999;
    const unsigned int order = 100;
    const double box_len = 10.0;
    const unsigned int M = 100;
    const unsigned int N = 100;

    // Create random particle positions
    std::vector<std::vector<double>> particle_positions(num_particles, std::vector<double>(3));
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto &pos : particle_positions)
    {
        pos[0] = dis(gen);
        pos[1] = dis(gen);
        pos[2] = dis(gen);
    }

    std::vector<std::vector<double>> result = calculate_structure_factor(particle_positions, order, box_len, M, N);
    return 0;
}
```

### In Python

After running the setup script, you can import and use EspressoSq in your Python code:
```python
import sq_avx
result = sq_avx.calculate_structure_factor(particle_positions, order, box_len, M, N)

```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License


## Contact

