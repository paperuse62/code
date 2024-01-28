#ifndef CONFIG_HPP
#define CONFIG_HPP

// QIR backend: {CPU, NVGPU, AMDGPU}

// #define CPU
// #define NVGPU
// #define AMDGPU

// Track per circuit execution performance
#define PRINT_MEA_PER_CIRCUIT

// Error check for all NVIDIA CUDA Runtim-API calls and Kernel check
#define CUDA_ERROR_CHECK

// Error check for all AMD HIP Runtim-API calls and Kernel check
#define HIP_ERROR_CHECK

// Accelerate by AVX512
// #define USE_AVX512

// ================================= Configurations =====================================
namespace SVSim
{
    // Basic Type for indices, adjust to uint64_t when qubits > 15
    using IdxType = unsigned long;
    // Basic Type for value, expect to support half, float and double
    using ValType = double;
// Random seed
#define RAND_SEED time(0)
// Tile for transposition in the adjoint operation
#define TILE 16
// Threads per GPU Thread BLock (Fixed)
#define THREADS_PER_BLOCK 256
// Error bar for validation
#define ERROR_BAR (1e-3)
// constant value of PI
#define PI 3.14159265358979323846
// constant value of 1/sqrt(2)
#define S2I 0.70710678118654752440
//// avx bitwidth for CPU
#define AVX 512
//// vector length
#define VEC ((AVX) / sizeof(IdxType))

#define OUTER_SIZE 2
#define INNER_SIZE 18

    const IdxType bitsetSize = 50;

}; // namespace SVSim

#endif
