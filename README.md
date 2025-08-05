# EspressoSq

**EspressoSq** is a high-performance library for structure factor calculations, implemented in C++ with SIMD optimizations and physics-based improvements. It includes a setup script to generate a Python wrapper using Cython, making it accessible for both C++ and Python users.

---

## Features

* **Fast Calculations**: Optimized with SIMD instructions for superior speed.
* **Physics-Based Optimizations**:
  Choose how many wavevectors to use for sampling and how many `q`s to calculate. The library automatically distributes them uniformly on a logarithmic scale.
* **Cross-Platform**: C++ core with a Python wrapper for ease of use in Python environments.
* **Minimal Dependencies**:
  I suggest using **GCC** for compilation, but otherwise the library is dependency-free.

---

## ⚙️ Installation

### 🔧 Prerequisites

* A working **GCC compiler**
* **Python** with **Cython** installed

### 📦 Build Steps

1. **Clone the repository**

2. **Create a build directory and compile the library**

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. **(Optional) Build Python wrapper**
   To make the package importable in Python:

   ```bash
   python3 setup.py build_ext --inplace
   ```

   > ℹ️ You may need to install `setuptools`:

   ```bash
   pip3 install setuptools
   ```

---

## 🧪 Usage

### 🔬 In C++

Include the header file and link against the compiled library:

```cpp
#include "sq_avx.hpp"

int main() {
    const unsigned int num_particles = 9999;
    const unsigned int order = 100;
    const double box_len = 10.0;
    const unsigned int M = 100;
    const unsigned int N = 100;

    // Create random particle positions
    std::vector<std::vector<double>> particle_positions(num_particles, std::vector<double>(3));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, box_len);

    for (auto &pos : particle_positions) {
        pos[0] = dis(gen);
        pos[1] = dis(gen);
        pos[2] = dis(gen);
    }

    auto result = calculate_structure_factor(particle_positions, order, box_len, M, N);
    return 0;
}
```

---

### 🐍 In Python

After building the Python extension, you can use it like this:

```python
import sq_avx

# particle_positions should be a list of 3D positions
result = sq_avx.calculate_structure_factor(particle_positions, order, box_len, M, N)
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/stekajack/espressoSq).

