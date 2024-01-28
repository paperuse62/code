# BML-QSim: Quantum Simulation Framework

Welcome to BML-QSim, a cutting-edge quantum simulation framework uniquely designed for heterogeneous computing platforms. BML-QSim stands out in the realm of quantum simulations by significantly increasing the supported number of qubits while maintaining exceptional performance. This breakthrough is achieved through a meticulously designed pipeline that optimally leverages the capabilities of various computing architectures.

## Prerequisites

Before you begin, ensure you have the following environment setup:

- CUDA (version 12.3.107 in our test)
- GCC (version 9.4.0 in our test)
- [NVComp](https://github.com/NVIDIA/nvcomp)
- OpenMP
- Python
- GPUDirect Storage

## Installation

### Setting Up

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/paperuse62/code.git BML-QSim
   cd BML-QSim
   ```

2. **Configure SSD Path:**
   Modify `bmlqsim.cuh` at line 412 to set up the SSD path according to your system configuration.

3. **Update NVComp Path:**
   Edit `compile.py` to specify the correct path of your NVComp installation. NVComp can be found [here](https://github.com/NVIDIA/nvcomp).

### Converting QASM to CU Files

To convert QASM files to .cu files, use the `qasm_to_c.py` script:

```bash
python qasm_to_c.py -i /path/to/qasm/file -o /path/to/cu/file
```

This script facilitates the translation of quantum assembly language into CUDA code.

### Compilation

Run the following command to compile the executables:

```bash
python compile.py /path/to/cu/file /path/to/executable
```

Ensure that all dependencies are correctly installed and paths are properly set as mentioned above.

## Usage

To run single test:
```bash
/path/to/executable
```

To run the end-to-end benchmark process, use the provided example script:

```bash
python benchmark.py
```

This script will guide you through the entire benchmarking process, showcasing the capabilities of BML-QSim.

## Acknowledgements

BML-QSim is built upon the foundation of SV-Sim. For more information about SV-Sim, visit their [repository](https://github.com/pnnl/SV-Sim).
