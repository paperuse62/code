#ifndef UTIL_NVGPU_CUH
#define UTIL_NVPUG_CUH

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#include "config.hpp"

namespace SVSim
{

//==================================== Error Checking =======================================
// Error checking for CUDA API call
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
    inline void __cudaSafeCall(cudaError err, const char *file, const int line)
    {
#ifdef CUDA_ERROR_CHECK
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
#endif
        return;
    }

// Error checking for CUDA API call
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)
    inline void __cudaCheckError(const char *file, const int line)
    {
#ifdef CUDA_ERROR_CHECK
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
        // Expensive checking
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
            exit(-1);
        }
#endif
        return;
    }

// Checking null pointer
#define CHECK_NULL_POINTER(X) __checkNullPointer(__FILE__, __LINE__, (void **)&(X))
    inline void __checkNullPointer(const char *file, const int line, void **ptr)
    {
        if ((*ptr) == NULL)
        {
            fprintf(stderr, "Error: NULL pointer at %s:%i.\n", file, line);
            exit(-1);
        }
    }

// Macro to wrap CUDA API calls and check for errors
#define CHECK_BITCOMP(call)                                         \
    do                                                              \
    {                                                               \
        bitcompResult_t err = call;                                 \
        if (err != BITCOMP_SUCCESS)                                 \
        {                                                           \
            fprintf(stderr, "BITCOMP error in %s at line %d: %d\n", \
                    __FILE__, __LINE__, err);                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

//================================= Allocation and Free ====================================
// CPU host allocation
#define SAFE_ALOC_HOST(X, Y) cudaSafeCall(cudaMallocHost((void **)&(X), (Y)));
// GPU device allocation
#define SAFE_ALOC_GPU(X, Y) cudaSafeCall(cudaMalloc((void **)&(X), (Y)));
// CPU host free
#define SAFE_FREE_HOST(X)                \
    if ((X) != NULL)                     \
    {                                    \
        cudaSafeCall(cudaFreeHost((X))); \
        (X) = NULL;                      \
    }
// GPU device free
#define SAFE_FREE_GPU(X)             \
    if ((X) != NULL)                 \
    {                                \
        cudaSafeCall(cudaFree((X))); \
        (X) = NULL;                  \
    }

    //======================================== Timer ==========================================
    double get_cpu_timer()
    {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        // get current timestamp in milliseconds
        return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
    }

    // CPU Timer object definition
    typedef struct CPU_Timer
    {
        CPU_Timer() { start = stop = 0.0; }
        void start_timer() { start = get_cpu_timer(); }
        void stop_timer() { stop = get_cpu_timer(); }
        double measure()
        {
            double millisconds = stop - start;
            return millisconds;
        }
        double start;
        double stop;
    } cpu_timer;

    // GPU Timer object definition
    typedef struct GPU_Timer
    {
        GPU_Timer()
        {
            cudaSafeCall(cudaEventCreate(&this->start));
            cudaSafeCall(cudaEventCreate(&this->stop));
        }
        ~GPU_Timer()
        {
            cudaSafeCall(cudaEventDestroy(this->start));
            cudaSafeCall(cudaEventDestroy(this->stop));
        }
        void start_timer() { cudaSafeCall(cudaEventRecord(this->start)); }
        void stop_timer() { cudaSafeCall(cudaEventRecord(this->stop)); }
        double measure()
        {
            cudaSafeCall(cudaEventSynchronize(this->stop));
            float millisconds = 0;
            cudaSafeCall(cudaEventElapsedTime(&millisconds, this->start, this->stop));
            return (double)millisconds;
        }
        cudaEvent_t start;
        cudaEvent_t stop;
    } gpu_timer;

    //======================================== Other ==========================================
    // Swap two pointers
    inline void swap_pointers(ValType **pa, ValType **pb)
    {
        ValType *tmp = (*pa);
        (*pa) = (*pb);
        (*pb) = tmp;
    }
    // // Verify whether a number is power of 2
    // inline bool is_power_of_2(int x)
    // {
    //     return (x > 0 && !(x & (x - 1)));
    // }

    size_t getFileSize(const std::string filename)
    {
        std::cout << "file name: " << filename << std::endl;
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return static_cast<size_t>(in.tellg());
    }

    template <typename T>
    void read_binary_to_array(const std::string &fname, T *_a, size_t dtype_len)
    {
        std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
        if (not ifs.is_open())
        {
            std::cerr << "fail to open " << fname << std::endl;
            exit(1);
        }
        ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ifs.close();
    }

    template <typename T>
    void write_array_to_binary(const std::string &fname, T *const _a, size_t const dtype_len)
    {
        std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
        if (not ofs.is_open())
            return;
        ofs.write(reinterpret_cast<const char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ofs.close();
    }

    void cufileWrite(const char *dir, void *inputPtrDevice, const size_t size)
    {
        CUfileDescr_t cf_descr;
        CUfileError_t status;
        CUfileHandle_t cf_handle;
        ssize_t ret = -1;
        int fd;

        // Write loaded data from GPU memory to a new file
        fd = open(dir, O_CREAT | O_RDWR | O_DIRECT, 0664);
        if (fd < 0)
        {
            std::cerr << "write file open error : " << std::strerror(errno) << std::endl;
            return;
        }

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS)
        {
            std::cerr << "file register error" << std::endl;
            close(fd);
            return;
        }

        ret = cuFileWrite(cf_handle, inputPtrDevice, size, 0, 0);
        if (ret < 0)
        {
            if (IS_CUFILE_ERR(ret))
                std::cerr << "write failed" << std::endl;
        }

        cuFileHandleDeregister(cf_handle);
        close(fd);
    }

    void cufileRead(const char *dir, void *outputPtrDevice, const size_t size)
    {
        CUfileDescr_t cf_descr;
        CUfileError_t status;
        CUfileHandle_t cf_handle;
        ssize_t ret = -1;
        int fd;

        // Write loaded data from GPU memory to a new file
        fd = open(dir, O_RDONLY | O_DIRECT);
        if (fd < 0)
        {
            std::cerr << "read file open error : " << dir << std::strerror(errno) << std::endl;
            return;
        }

        memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
        cf_descr.handle.fd = fd;
        cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS)
        {
            std::cerr << "file register error" << std::endl;
            close(fd);
            return;
        }

        ret = cuFileRead(cf_handle, outputPtrDevice, size, 0, 0);
        if (ret < 0)
        {
            if (IS_CUFILE_ERR(ret))
                std::cerr << "read failed" << std::endl;
        }

        cuFileHandleDeregister(cf_handle);
        close(fd);
    }

    __global__ void log2Transform(double *input, IdxType n, int8_t *signArr)
    {
        IdxType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            if (input[idx] < 0)
            {
                signArr[idx] = 1;
                input[idx] = log2(-input[idx]);
            }
            else if (input[idx] > 0)
            {
                signArr[idx] = 0;
                input[idx] = log2(input[idx]);
            }
            else if (input[idx] == 0)
            {
                signArr[idx] = 2;
                input[idx] = 0;
            }
        }
    }

    __global__ void exp2IntTransform(double *input, IdxType n, int8_t *signArr)
    {
        IdxType idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            if (signArr[idx] == 1)
            {
                input[idx] = -pow(2.0, input[idx]);
            }
            else if (signArr[idx] == 0)
            {
                input[idx] = pow(2.0, input[idx]);
            }
            else if (signArr[idx] == 2)
            {
                input[idx] = 0;
            }
        }
    }

    __host__ __device__ IdxType exp2Int(IdxType exp)
    {
        IdxType res = pow(2, exp);
        return res;
    }

    void increment(std::bitset<bitsetSize> &bitset, IdxType *indexArr, IdxType indexArrSize)
    {
        for (int i = 0; i < indexArrSize; i++)
        {
            if (bitset[indexArr[i]] == 0)
            {
                bitset[indexArr[i]] = 1;
                break;
            }
            else
            {
                bitset[indexArr[i]] = 0;
            }
        }
        // std::cout << bitset.to_ulong() << std::endl;
    }

}; // namespace SVSim

#endif
