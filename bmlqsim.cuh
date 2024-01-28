#ifndef SVSIM_NVGPU_OMP_CUH
#define svSIM_NVGPU_OMP_CUH

// #define CIRCUIT_PARTITION_PRINT
#define ENABLE_COMPRESSION
#define PWR_ERROR_BOUND 1e-3

#include <assert.h>
#include <cooperative_groups.h>
#include <vector>
#include <omp.h>
#include <sstream>
#include <string>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <bitset>
#include <algorithm>
#include <chrono>
#include <set>
#include <numeric>
#include <cmath>
#include <fstream>

#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "cufile.h"

#include <sys/sysinfo.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "config.hpp"
#include "utils.cuh"
#include <native/bitcomp.h>

namespace SVSim
{

    using namespace cooperative_groups;
    using namespace std;
    class Gate;
    class Simulation;
    using func_t = void (*)(const Gate *, const Simulation *, ValType **, ValType **, IdxType, IdxType);

    // Simulation runtime, is_forward?
    __global__ void simulation_kernel(Simulation *, IdxType gpuIdx, IdxType start, IdxType end, IdxType simulationOffset);

    // Current svSim supported gates: 38
    enum OP
    {
        U3,
        U2,
        U1,
        CX,
        ID,
        X,
        Y,
        Z,
        H,
        S,
        SDG,
        T,
        TDG,
        RX,
        RY,
        RZ,
        CZ,
        CY,
        SWAP,
        CH,
        CCX,
        CSWAP,
        CRX,
        CRY,
        CRZ,
        CU1,
        CU3,
        RXX,
        RZZ,
        RCCX,
        RC3X,
        C3X,
        C3SQRTX,
        C4X,
        R,
        SRN,
        W,
        RYY
    };

    // Name of the gate for tracing purpose
    const char *OP_NAMES[] = {
        "U3", "U2", "U1", "CX", "ID", "X", "Y", "Z", "H", "S",
        "SDG", "T", "TDG", "RX", "RY", "RZ", "CZ", "CY", "SWAP", "CH",
        "CCX", "CSWAP", "CRX", "CRY", "CRZ", "CU1", "CU3", "RXX", "RZZ", "RCCX",
        "RC3X", "C3X", "C3SQRTX", "C4X", "R", "SRN", "W", "RYY"};

    // Define gate function pointers
    extern __device__ func_t pU3_OP;
    extern __device__ func_t pU2_OP;
    extern __device__ func_t pU1_OP;
    extern __device__ func_t pCX_OP;
    extern __device__ func_t pID_OP;
    extern __device__ func_t pX_OP;
    extern __device__ func_t pY_OP;
    extern __device__ func_t pZ_OP;
    extern __device__ func_t pH_OP;
    extern __device__ func_t pS_OP;
    extern __device__ func_t pSDG_OP;
    extern __device__ func_t pT_OP;
    extern __device__ func_t pTDG_OP;
    extern __device__ func_t pRX_OP;
    extern __device__ func_t pRY_OP;
    extern __device__ func_t pRZ_OP;
    extern __device__ func_t pCZ_OP;
    extern __device__ func_t pCY_OP;
    extern __device__ func_t pSWAP_OP;
    extern __device__ func_t pCH_OP;
    extern __device__ func_t pCCX_OP;
    extern __device__ func_t pCSWAP_OP;
    extern __device__ func_t pCRX_OP;
    extern __device__ func_t pCRY_OP;
    extern __device__ func_t pCRZ_OP;
    extern __device__ func_t pCU1_OP;
    extern __device__ func_t pCU3_OP;
    extern __device__ func_t pRXX_OP;
    extern __device__ func_t pRZZ_OP;
    extern __device__ func_t pRCCX_OP;
    extern __device__ func_t pRC3X_OP;
    extern __device__ func_t pC3X_OP;
    extern __device__ func_t pC3SQRTX_OP;
    extern __device__ func_t pC4X_OP;
    extern __device__ func_t pR_OP;
    extern __device__ func_t pSRN_OP;
    extern __device__ func_t pW_OP;
    extern __device__ func_t pRYY_OP;

    // Gate definition, currently support up to 5 qubit indices and 3 rotation params
    class Gate
    {
    public:
        Gate(enum OP _op_name,
             IdxType _qb0, IdxType _qb1, IdxType _qb2, IdxType _qb3, IdxType _qb4,
             ValType _theta, ValType _phi, ValType _lambda) : op_name(_op_name),
                                                              qb0(_qb0), qb1(_qb1), qb2(_qb2), qb3(_qb3), qb4(_qb4),
                                                              theta(_theta), phi(_phi), lambda(_lambda) {}

        ~Gate() {}

        // upload to a specific GPU
        Gate *upload(IdxType dev)
        {
            cudaSafeCall(cudaSetDevice(dev));
            Gate *gpu;
            SAFE_ALOC_GPU(gpu, sizeof(Gate));

#define GATE_BRANCH(GATE)                                                      \
    case GATE:                                                                 \
        cudaSafeCall(cudaMemcpyFromSymbol(&op, p##GATE##_OP, sizeof(func_t))); \
        break;
            switch (op_name)
            {
                GATE_BRANCH(U3);
                GATE_BRANCH(U2);
                GATE_BRANCH(U1);
                GATE_BRANCH(CX);
                GATE_BRANCH(ID);
                GATE_BRANCH(X);
                GATE_BRANCH(Y);
                GATE_BRANCH(Z);
                GATE_BRANCH(H);
                GATE_BRANCH(S);
                GATE_BRANCH(SDG);
                GATE_BRANCH(T);
                GATE_BRANCH(TDG);
                GATE_BRANCH(RX);
                GATE_BRANCH(RY);
                GATE_BRANCH(RZ);
                GATE_BRANCH(CZ);
                GATE_BRANCH(CY);
                GATE_BRANCH(SWAP);
                GATE_BRANCH(CH);
                GATE_BRANCH(CCX);
                GATE_BRANCH(CSWAP);
                GATE_BRANCH(CRX);
                GATE_BRANCH(CRY);
                GATE_BRANCH(CRZ);
                GATE_BRANCH(CU1);
                GATE_BRANCH(CU3);
                GATE_BRANCH(RXX);
                GATE_BRANCH(RZZ);
                GATE_BRANCH(RCCX);
                GATE_BRANCH(RC3X);
                GATE_BRANCH(C3X);
                GATE_BRANCH(C3SQRTX);
                GATE_BRANCH(C4X);
                GATE_BRANCH(R);
                GATE_BRANCH(SRN);
                GATE_BRANCH(W);
                GATE_BRANCH(RYY);
            }
            cudaSafeCall(cudaMemcpy(gpu, this, sizeof(Gate), cudaMemcpyHostToDevice));
            return gpu;
        }
        // applying the embedded gate operation on GPU side
        __device__ void exe_op(Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
        {
            (*(this->op))(this, sim, svRealPtrArrDevice, svImagPtrArrDeivce, gpuIdx, simulationOffset);
        }
        // dump the current circuit
        void dump(std::stringstream &ss)
        {
            ss << OP_NAMES[op_name] << "(" << qb0 << "," << qb1 << ","
               << qb2 << "," << qb3 << ","
               << qb4 << "," << theta << ","
               << phi << "," << lambda << ");" << std::endl;
        }
        // Gate operation
        func_t op;
        // Gate name
        enum OP op_name;
        // Qubit position parameters
        IdxType qb0;
        IdxType qb1;
        IdxType qb2;
        IdxType qb3;
        IdxType qb4;
        // Qubit rotation parameters
        ValType theta;
        ValType phi;
        ValType lambda;
    }; // end of Gate definition

    class Simulation
    {
    public:
        Simulation(IdxType _numQubits, IdxType _numGpus, IdxType _chunkSize)
            : numQubits(_numQubits),
              numGpus(_numGpus),
              dim((IdxType)1 << (numQubits)),
              half_dim((IdxType)1 << (numQubits - 1)),
              gpu_mem(0),
              n_gates(0),
              // gpu_scale is the logarithm of the number of GPUs
              gpu_scale(floor(log((double)_numGpus + 0.5) / log(2.0))),
              lg2_m_gpu(numQubits - gpu_scale),
              m_gpu((IdxType)1 << (lg2_m_gpu)),
              svSize(dim * (IdxType)sizeof(ValType)),
              svSizePerGpu(svSize / numGpus),
              circuit_gpu(NULL),
              sim_gpu(NULL),
              chunkSize(_chunkSize),
              chunkNum(1 << (numQubits - chunkSize))
        {
            struct sysinfo info;
            if (sysinfo(&info) != 0)
            {
                std::cerr << "Error: system info grab failed." << std::endl;
                exit(1);
            }

            std::cout << "CPU free memory: " << info.freeram / 1024 / 1024 / 1024 << " GB" << std::endl;

            CUfileError_t status;
            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                status = cuFileDriverOpen();
                if (status.err != CU_FILE_SUCCESS)
                {
                    std::cerr << " cuFile driver failed to open " << std::endl;
                    return;
                }
            }

            streamNumPerGpu = 2;
            streamNumPerGpuLog2 = 1;

            // the sign array allocation

            SAFE_ALOC_HOST(svRealSignPtrHost, exp2Int(numQubits - chunkSize) * sizeof(int8_t *));
            SAFE_ALOC_HOST(svImagSignPtrHost, exp2Int(numQubits - chunkSize) * sizeof(int8_t *));

            for (int i = 0; i < exp2Int(numQubits - chunkSize); i++)
            {
                SAFE_ALOC_HOST(svRealSignPtrHost[i], sizeof(int8_t) * exp2Int(chunkSize));
                SAFE_ALOC_HOST(svImagSignPtrHost[i], sizeof(int8_t) * exp2Int(chunkSize));
                memset(svRealSignPtrHost[i], 0, exp2Int(chunkSize) * sizeof(int8_t));
                memset(svImagSignPtrHost[i], 0, exp2Int(chunkSize) * sizeof(int8_t));
            }

            // gpu side initialization

            SAFE_ALOC_HOST(svRealPtrArrDevice, sizeof(ValType *) * numGpus);
            SAFE_ALOC_HOST(svImagPtrArrDeivce, sizeof(ValType *) * numGpus);

            SAFE_ALOC_HOST(circuitCopy, sizeof(vector<Gate *> *) * numGpus);

            SAFE_ALOC_HOST(compressedSizeRealDevice, sizeof(size_t *) * numGpus);
            SAFE_ALOC_HOST(compressedSizeImagDevice, sizeof(size_t *) * numGpus);

            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                SAFE_ALOC_GPU(compressedSizeRealDevice[d], sizeof(size_t) * exp2Int(numQubits - chunkSize));
                SAFE_ALOC_GPU(compressedSizeImagDevice[d], sizeof(size_t) * exp2Int(numQubits - chunkSize));
            }

            SAFE_ALOC_HOST(chunkLocationRealArr, sizeof(int8_t) * chunkNum);
            SAFE_ALOC_HOST(chunkLocationImagArr, sizeof(int8_t) * chunkNum);
            memset(chunkLocationRealArr, 0, sizeof(int8_t) * chunkNum);
            memset(chunkLocationImagArr, 0, sizeof(int8_t) * chunkNum);

            bitcompHandle_t plan;
            bitcompCreatePlan(
                &plan,                                // Bitcomp handle
                sizeof(ValType) * exp2Int(chunkSize), // Size in bytes of the uncompressed data
                BITCOMP_FP64_DATA,                    // Data type
                BITCOMP_LOSSY_FP_TO_SIGNED,           // Compression type
                BITCOMP_DEFAULT_ALGO);                // Bitcomp algo, default or sparse

            IdxType maxlen = bitcompMaxBuflen(sizeof(ValType) * exp2Int(chunkSize));

#ifdef ENABLE_COMPRESSION
            std::chrono::time_point<std::chrono::system_clock> start, end;
            std::chrono::duration<double> time;

            for (int i = 0; i < chunkNum; i++)
            {
                for (int j = 0; j < exp2Int(chunkSize); j++)
                {
                    svRealSignPtrHost[i][j] = 2;
                    svImagSignPtrHost[i][j] = 2;
                }
            }
            svRealSignPtrHost[0][0] = 0;

            double pwrErrorBound = PWR_ERROR_BOUND;
            double absErrorBound = log2(1.0 + pwrErrorBound);

            ValType *synthesisData;
            ValType *synthesisDataComped;

            SAFE_ALOC_HOST(synthesisData, sizeof(ValType) * exp2Int(chunkSize));
            SAFE_ALOC_HOST(synthesisDataComped, sizeof(ValType) * exp2Int(chunkSize));

            for (int i = 0; i < exp2Int(chunkSize); i++)
            {
                synthesisData[i] = 0;
            }

            start = std::chrono::system_clock::now();

            CHECK_BITCOMP(bitcompHostCompressLossy_fp64(plan, synthesisData, synthesisDataComped, absErrorBound));

            end = std::chrono::system_clock::now();

            size_t compedSize = 0;

            bitcompGetCompressedSize(synthesisDataComped, &compedSize);

            // the main memory compressed chunk allocation

            SAFE_ALOC_HOST(svRealCompressedPtrArrHost, sizeof(ValType *) * exp2Int(numQubits - chunkSize));
            SAFE_ALOC_HOST(svImagCompressedPtrArrHost, sizeof(ValType *) * exp2Int(numQubits - chunkSize));

            SAFE_ALOC_HOST(svRealCompressedSizePtrArrHost, sizeof(size_t) * chunkNum);
            SAFE_ALOC_HOST(svImagCompressedSizePtrArrHost, sizeof(size_t) * chunkNum);

            for (int i = 0; i < chunkNum; i++)
            {
                SAFE_ALOC_HOST(svRealCompressedPtrArrHost[i], compedSize);
                SAFE_ALOC_HOST(svImagCompressedPtrArrHost[i], compedSize);
                memcpy(svRealCompressedPtrArrHost[i], synthesisDataComped, compedSize);
                memcpy(svImagCompressedPtrArrHost[i], synthesisDataComped, compedSize);
                svRealCompressedSizePtrArrHost[i] = compedSize;
                svImagCompressedSizePtrArrHost[i] = compedSize;
            }

            SAFE_FREE_HOST(synthesisData);
            SAFE_FREE_HOST(synthesisDataComped);

            ssdPath = "/path/to/ssd";

            time = end - start;
            std::cout << "Initial compression time: "
                      << time.count()
                      << " s "
                      << std::endl;
#endif

            SAFE_ALOC_HOST(numBlocksPerSm, sizeof(int) * numGpus);

            SAFE_ALOC_HOST(deviceProp, sizeof(cudaDeviceProp) * numGpus);
        }

        ~Simulation()
        {
            // Release circuit
            clear_circuit();

            // Release for GPU side
            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                SAFE_FREE_GPU(svRealPtrArrDevice[d]);
                SAFE_FREE_GPU(svImagPtrArrDeivce[d]);
            }

            SAFE_FREE_HOST(svRealPtrArrDevice);
            SAFE_FREE_HOST(svImagPtrArrDeivce);

            for (int i = 0; i < exp2Int(numQubits - chunkSize); i++)
            {
                SAFE_FREE_HOST(svRealCompressedPtrArrHost[i]);
                SAFE_FREE_HOST(svImagCompressedPtrArrHost[i]);
            }

            SAFE_FREE_HOST(svRealCompressedPtrArrHost);
            SAFE_FREE_HOST(svImagCompressedPtrArrHost);

            SAFE_FREE_HOST(svRealCompressedSizePtrArrHost);
            SAFE_FREE_HOST(svImagCompressedSizePtrArrHost);

            SAFE_FREE_HOST(circuitCopy);

            SAFE_FREE_HOST(numBlocksPerSm);

            SAFE_FREE_HOST(deviceProp);

            SAFE_FREE_HOST(chunkLocationRealArr);
            SAFE_FREE_HOST(chunkLocationImagArr);

            for (unsigned d = 0; d < numGpus; d++)
            {
                // cout << "Closing File Driver." << std::endl;
                (void)cuFileDriverClose();
            }

            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                SAFE_FREE_GPU(compressedSizeRealDevice[d]);
                SAFE_FREE_GPU(compressedSizeImagDevice[d]);
            }

            SAFE_FREE_HOST(compressedSizeRealDevice);
            SAFE_FREE_HOST(compressedSizeImagDevice);

            for (int i = 0; i < exp2Int(numQubits - chunkSize); i++)
            {
                SAFE_FREE_HOST(svRealSignPtrHost[i]);
                SAFE_FREE_HOST(svImagSignPtrHost[i]);
            }

            SAFE_FREE_HOST(svRealSignPtrHost);
            SAFE_FREE_HOST(svImagSignPtrHost);
        }

        // add a gate to the current circuit
        void append(Gate *g)
        {
            CHECK_NULL_POINTER(g);
            assert((g->qb0 < numQubits));
            assert((g->qb1 < numQubits));
            assert((g->qb2 < numQubits));
            assert((g->qb3 < numQubits));
            assert((g->qb4 < numQubits));

            // Be careful! PyBind11 will auto-release the object pointed by g,
            // so we need to creat a new Gate object inside the code
            circuit.push_back(new Gate(*g));
            n_gates++;
        }

        // tell if a idx is in the an array
        bool isInArray(IdxType *arr, IdxType arrSize, IdxType idx)
        {
            for (int i = 0; i < arrSize; i++)
            {
                if (arr[i] == idx)
                {
                    return true;
                }
            }
            return false;
        }

        // get the index of a value in an array
        IdxType getIndex(IdxType *arr, IdxType arrSize, IdxType idx)
        {
            for (int i = 0; i < arrSize; i++)
            {
                if (arr[i] == idx)
                {
                    return i;
                }
            }
            return 10000000;
        }

        // mapping the index
        IdxType idxMapping(IdxType idx)
        {
            if (isInArray(globalInnerArr, globalInnerArrSize, idx))
            {
                return chunkSize + getIndex(globalInnerArr, globalInnerArrSize, idx);
            }
            else
            {
                return idx;
            }
        }

        Simulation *upload(IdxType start, IdxType end)
        {
            assert(n_gates == circuit.size());
            // Should be null after calling clear_circuit()
            assert(circuit_gpu == NULL);
            assert(sim_gpu == NULL);

            for (IdxType t = start; t < end; t++)
            {
                // set the new gate qubit index based on the inner index
                circuit[t]->qb0 = idxMapping(circuit[t]->qb0);
                circuit[t]->qb1 = idxMapping(circuit[t]->qb1);
                circuit[t]->qb2 = idxMapping(circuit[t]->qb2);
                circuit[t]->qb3 = idxMapping(circuit[t]->qb3);
                circuit[t]->qb4 = idxMapping(circuit[t]->qb4);
            }

            SAFE_ALOC_HOST(sim_gpu, sizeof(Simulation *) * numGpus);
            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                for (IdxType t = start; t < end; t++)
                {
                    // circuit[t]->dump();
                    Gate *g_gpu = circuit[t]->upload(d);
                    circuitCopy[d].push_back(g_gpu);
                }
                SAFE_ALOC_GPU(circuit_gpu, n_gates * sizeof(Gate *));
                cudaSafeCall(cudaMemcpy(circuit_gpu, circuitCopy[d].data(),
                                        n_gates * sizeof(Gate *), cudaMemcpyHostToDevice));

                SAFE_ALOC_GPU(sim_gpu[d], sizeof(Simulation));
                cudaSafeCall(cudaMemcpy(sim_gpu[d], this,
                                        sizeof(Simulation), cudaMemcpyHostToDevice));
            }
            return this;
        }

        void cleanUploaded(IdxType start, IdxType end)
        {
            if (sim_gpu != NULL)
            {
                for (unsigned d = 0; d < numGpus; d++)
                {
                    SAFE_FREE_GPU(sim_gpu[d]);
                    for (unsigned i = 0; i < end - start; i++)
                        SAFE_FREE_GPU(circuitCopy[d][i]);
                    circuitCopy[d].clear();
                }
            }
            SAFE_FREE_HOST(sim_gpu);
            SAFE_FREE_GPU(circuit_gpu);
        }

        // dump the circuit
        std::string dump(IdxType start, IdxType end)
        {
            stringstream ss;
            for (IdxType t = start; t < end; t++)
            {
                circuit[t]->dump(ss);
            }
            return ss.str();
        }

        void getProperties()
        {
            for (int tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                cudaSafeCall(cudaSetDevice(tmpIdx));
                cudaSafeCall(cudaGetDeviceProperties(&deviceProp[tmpIdx], tmpIdx));
                cudaSafeCall(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm[tmpIdx], simulation_kernel, THREADS_PER_BLOCK, 0));
            }
        }

        // start sv simulation
        void bmlqsim(IdxType start, IdxType end, IdxType simulationOffset, IdxType gpuIdx, cudaStream_t *curStream)
        {
            IdxType d = gpuIdx;
            cudaSafeCall(cudaSetDevice(d));

            dim3 gridDim(1, 1, 1);
            gridDim.x = numBlocksPerSm[d] * deviceProp[d].multiProcessorCount;

            void *args[] = {&(sim_gpu[d]), &d, &start, &end, &simulationOffset};

            cudaLaunchCooperativeKernel((void *)simulation_kernel, gridDim, dim3(THREADS_PER_BLOCK), args, 0, *curStream);
            cudaCheckError();
            return;
        }

        void clear_circuit()
        {
            if (sim_gpu != NULL)
            {
                for (unsigned d = 0; d < numGpus; d++)
                {
                    SAFE_FREE_GPU(sim_gpu[d]);
                    for (unsigned i = 0; i < n_gates; i++)
                        SAFE_FREE_GPU(circuitCopy[d][i]);
                    circuitCopy[d].clear();
                }
            }
            for (unsigned i = 0; i < n_gates; i++)
            {
                delete circuit[i];
            }
            SAFE_FREE_HOST(sim_gpu);
            SAFE_FREE_GPU(circuit_gpu);
            circuit.clear();
            n_gates = 0;
        }

        // clear index
        void clearIdx()
        {
            globalOuterArrSize = 0;
            globalInnerArrSize = 0;
            free(globalOuterArr);
            free(globalInnerArr);
        }

        void getMemInfo(size_t *free_byte, size_t *total_byte)
        {
            cudaError_t cuda_status = cudaMemGetInfo(free_byte, total_byte);
            if (cudaSuccess != cuda_status)
            {
                std::cerr << "Error: " << cudaGetErrorString(cuda_status) << std::endl;
                return;
            }
        }

        void getMaxChunkNum()
        {
            size_t freeByte;
            size_t totalByte;
            size_t freeByteArr[numGpus] = {};

            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                cudaSetDevice(tmpIdx);
                getMemInfo(&freeByte, &totalByte);
                freeByteArr[tmpIdx] = freeByte;
            }

            size_t *minFreeByte = std::min_element(freeByteArr, freeByteArr + numGpus);
            float freeGB = *minFreeByte / 1024.0 / 1024.0 / 1024.0;
            std::cout << "the minimum free memory among GPUs is " << freeGB << " GB" << std::endl;
            size_t maxChunkNumPerGpu = *minFreeByte >> (chunkSize + 4 + 2);
            maxGlobalInnerNum = static_cast<IdxType>(log2(static_cast<double>(maxChunkNumPerGpu)));
            size_t logNumGpus = floor(log((double)numGpus + 0.5) / log(2.0));
            size_t logStreamNumPerGpu = floor(log((double)streamNumPerGpu + 0.5) / log(2.0));
            maxGlobalInnerNum = min(numQubits - chunkSize - logNumGpus - logStreamNumPerGpu, (unsigned long)maxGlobalInnerNum);

            std::cout << "maximum chunk number per gpu is " << maxChunkNumPerGpu << std::endl;
            std::cout << "maximum global inner number is " << maxGlobalInnerNum << std::endl;
        }

        void deviceBufferInit()
        {
            // GPU side initialization
            for (unsigned d = 0; d < numGpus; d++)
            {
                cudaSafeCall(cudaSetDevice(d));
                // GPU memory allocation
                SAFE_ALOC_GPU(svRealPtrArrDevice[d], sizeof(ValType) * exp2Int(maxGlobalInnerNum + chunkSize + streamNumPerGpuLog2));
                SAFE_ALOC_GPU(svImagPtrArrDeivce[d], sizeof(ValType) * exp2Int(maxGlobalInnerNum + chunkSize + streamNumPerGpuLog2));
                gpu_mem += svSizePerGpu * 2;
            }

            // initialize the buffers on device arr[gpuIndex][bufferIndex][dataIndex]
            deviceRealCompressedBufferArr = (ValType ***)malloc(sizeof(ValType **) * numGpus);
            deviceImagCompressedBufferArr = (ValType ***)malloc(sizeof(ValType **) * numGpus);
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                deviceRealCompressedBufferArr[tmpIdx] = (ValType **)malloc(sizeof(ValType *) * streamNumPerGpu);
                deviceImagCompressedBufferArr[tmpIdx] = (ValType **)malloc(sizeof(ValType *) * streamNumPerGpu);
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    cudaSetDevice(tmpIdx);
                    cudaMalloc((void **)&deviceRealCompressedBufferArr[tmpIdx][tmpIdx2], sizeof(ValType) * exp2Int(chunkSize + maxGlobalInnerNum));
                    cudaMalloc((void **)&deviceImagCompressedBufferArr[tmpIdx][tmpIdx2], sizeof(ValType) * exp2Int(chunkSize + maxGlobalInnerNum));
                }
            }

            // initialize the sign buffers on device
            deviceRealSignBufferArr = (int8_t ***)malloc(sizeof(int8_t **) * numGpus);
            deviceImagSignBufferArr = (int8_t ***)malloc(sizeof(int8_t **) * numGpus);
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                deviceRealSignBufferArr[tmpIdx] = (int8_t **)malloc(sizeof(int8_t *) * streamNumPerGpu);
                deviceImagSignBufferArr[tmpIdx] = (int8_t **)malloc(sizeof(int8_t *) * streamNumPerGpu);
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    cudaSetDevice(tmpIdx);
                    cudaMalloc((void **)&deviceRealSignBufferArr[tmpIdx][tmpIdx2], sizeof(int8_t) * exp2Int(chunkSize + maxGlobalInnerNum));
                    cudaMalloc((void **)&deviceImagSignBufferArr[tmpIdx][tmpIdx2], sizeof(int8_t) * exp2Int(chunkSize + maxGlobalInnerNum));
                }
            }
        }

        void deviceBufferFree()
        {
            // free the device compressed buffers
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    cudaSetDevice(tmpIdx);
                    cudaFree(deviceRealCompressedBufferArr[tmpIdx][tmpIdx2]);
                    cudaFree(deviceImagCompressedBufferArr[tmpIdx][tmpIdx2]);
                }
                free(deviceRealCompressedBufferArr[tmpIdx]);
                free(deviceImagCompressedBufferArr[tmpIdx]);
            }
            free(deviceRealCompressedBufferArr);
            free(deviceImagCompressedBufferArr);

            // free the device sign buffers
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    cudaSetDevice(tmpIdx);
                    cudaFree(deviceRealSignBufferArr[tmpIdx][tmpIdx2]);
                    cudaFree(deviceImagSignBufferArr[tmpIdx][tmpIdx2]);
                }
                free(deviceRealSignBufferArr[tmpIdx]);
                free(deviceImagSignBufferArr[tmpIdx]);
            }
        }

        // Function to calculate the sum of an array
        size_t sumArray(size_t *arr, size_t size)
        {
            size_t sum = 0;
            for (size_t i = 0; i < size; ++i)
            {
                sum += arr[i];
            }
            return sum;
        }

        void memq(IdxType start, IdxType end, IdxType stageIdx)
        {
            std::cout << "=================================================================== " << std::endl;

            auto e2eStart = std::chrono::high_resolution_clock::now();

            double pwrErrorBound = PWR_ERROR_BOUND;
            double absErrorBound = log2(1.0 + pwrErrorBound);

            for (IdxType tmpGpuIdx = 0; tmpGpuIdx < numGpus; tmpGpuIdx++)
            {
                cudaSetDevice(tmpGpuIdx);
                cudaMemset(compressedSizeRealDevice[tmpGpuIdx], 0, sizeof(size_t) * exp2Int(numQubits - chunkSize));
                cudaMemset(compressedSizeImagDevice[tmpGpuIdx], 0, sizeof(size_t) * exp2Int(numQubits - chunkSize));
            }

            bitset<bitsetSize> globalIndex(0);

            // Create CUDA streams
            cudaStream_t **streamArr = (cudaStream_t **)malloc(sizeof(cudaStream_t *) * numGpus);
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                cudaSetDevice(tmpIdx);
                streamArr[tmpIdx] = (cudaStream_t *)malloc(sizeof(cudaStream_t) * streamNumPerGpu);
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    cudaStreamCreate(&streamArr[tmpIdx][tmpIdx2]);
                }
            }

            // create the bitcomp plan and associate streams
            bitcompHandle_t **planArr = (bitcompHandle_t **)malloc(sizeof(bitcompHandle_t *) * numGpus);
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                planArr[tmpIdx] = (bitcompHandle_t *)malloc(sizeof(bitcompHandle_t) * streamNumPerGpu);
                for (IdxType tmpIdx2 = 0; tmpIdx2 < streamNumPerGpu; tmpIdx2++)
                {
                    bitcompCreatePlan(
                        &planArr[tmpIdx][tmpIdx2],            // Bitcomp handle
                        sizeof(ValType) * exp2Int(chunkSize), // Size in bytes of the uncompressed data
                        BITCOMP_FP64_DATA,                    // Data type
                        BITCOMP_LOSSY_FP_TO_SIGNED,           // Compression type
                        BITCOMP_DEFAULT_ALGO);                // Bitcomp algo, default or sparse
                    bitcompSetStream(planArr[tmpIdx][tmpIdx2], streamArr[tmpIdx][tmpIdx2]);
                }
            }

            // calculate the maximum compressed size for a given uncompressed size
            IdxType maxlen = bitcompMaxBuflen(sizeof(ValType) * exp2Int(chunkSize));

            // preparation for the main for loop
            IdxType gpuIdx = 0;
            IdxType streamIdx = 0;
            IdxType tmpOffsetIdx = 0;
            IdxType helperIdx = 0;

            for (IdxType tmpGlobalOuterIndex = 0; tmpGlobalOuterIndex < exp2Int(globalOuterArrSize); tmpGlobalOuterIndex++)
            {
#ifdef ENABLE_COMPRESSION
                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    if (chunkLocationRealArr[globalIndex.to_ulong() >> chunkSize] == 0)
                    {
                        cudaMemcpyAsync(deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize], svRealCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize], cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);
                    }
                    else if (chunkLocationRealArr[globalIndex.to_ulong() >> chunkSize] == 1)
                    {
                        std::string filePathStrReal = ssdPath + std::to_string(globalIndex.to_ulong() >> chunkSize) + ".real";
                        cufileRead(filePathStrReal.c_str(), deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]);
                    }

                    if (chunkLocationImagArr[globalIndex.to_ulong() >> chunkSize] == 0)
                    {
                        cudaMemcpyAsync(deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize], svImagCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize], cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);
                    }
                    else if (chunkLocationImagArr[globalIndex.to_ulong() >> chunkSize] == 1)
                    {
                        std::string filePathStrImag = ssdPath + std::to_string(globalIndex.to_ulong() >> chunkSize) + ".imag";
                        cufileRead(filePathStrImag.c_str(), deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]);
                    }

                    cudaMemcpyAsync(deviceRealSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealSignPtrHost[globalIndex.to_ulong() >> chunkSize], sizeof(int8_t) * exp2Int(chunkSize), cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);
                    cudaMemcpyAsync(deviceImagSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagSignPtrHost[globalIndex.to_ulong() >> chunkSize], sizeof(int8_t) * exp2Int(chunkSize), cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);

                    increment(globalIndex, globalInnerArr, globalInnerArrSize);
                }

                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    CHECK_BITCOMP(bitcompUncompress(planArr[gpuIdx][streamIdx], deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize)));
                    CHECK_BITCOMP(bitcompUncompress(planArr[gpuIdx][streamIdx], deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize)));
                }
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    exp2IntTransform<<<(exp2Int(chunkSize) + 255) / 256, 256, 0, streamArr[gpuIdx][streamIdx]>>>(svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), exp2Int(chunkSize), deviceRealSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize));
                    exp2IntTransform<<<(exp2Int(chunkSize) + 255) / 256, 256, 0, streamArr[gpuIdx][streamIdx]>>>(svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), exp2Int(chunkSize), deviceImagSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize));
                }

#else
                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    cudaMemcpyAsync(svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealPtrHost + globalIndex.to_ulong(), sizeof(ValType) * exp2Int(chunkSize), cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);
                    cudaMemcpyAsync(svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagPtrHost + globalIndex.to_ulong(), sizeof(ValType) * exp2Int(chunkSize), cudaMemcpyHostToDevice, streamArr[gpuIdx][streamIdx]);
                    increment(globalIndex, globalInnerArr, globalInnerArrSize);
                }
#endif

                bmlqsim(start, end, streamIdx, gpuIdx, &streamArr[gpuIdx][streamIdx]);

#ifdef ENABLE_COMPRESSION
                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    log2Transform<<<(exp2Int(chunkSize) + 255) / 256, 256, 0, streamArr[gpuIdx][streamIdx]>>>(svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), exp2Int(chunkSize), deviceRealSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize));
                    log2Transform<<<(exp2Int(chunkSize) + 255) / 256, 256, 0, streamArr[gpuIdx][streamIdx]>>>(svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), exp2Int(chunkSize), deviceImagSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize));
                }

                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    CHECK_BITCOMP(bitcompCompressLossy_fp64(planArr[gpuIdx][streamIdx], svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), absErrorBound));
                    CHECK_BITCOMP(bitcompCompressLossy_fp64(planArr[gpuIdx][streamIdx], svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), absErrorBound));

                    bitcompGetCompressedSizeAsync(deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), compressedSizeRealDevice[gpuIdx] + (globalIndex.to_ulong() >> chunkSize), streamArr[gpuIdx][streamIdx]);
                    bitcompGetCompressedSizeAsync(deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), compressedSizeImagDevice[gpuIdx] + (globalIndex.to_ulong() >> chunkSize), streamArr[gpuIdx][streamIdx]);

                    increment(globalIndex, globalInnerArr, globalInnerArrSize);
                }

                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    cudaMemcpyAsync(svRealCompressedSizePtrArrHost + (globalIndex.to_ulong() >> chunkSize), compressedSizeRealDevice[gpuIdx] + (globalIndex.to_ulong() >> chunkSize), sizeof(size_t), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);
                    cudaMemcpyAsync(svImagCompressedSizePtrArrHost + (globalIndex.to_ulong() >> chunkSize), compressedSizeImagDevice[gpuIdx] + (globalIndex.to_ulong() >> chunkSize), sizeof(size_t), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);

                    cudaStreamSynchronize(streamArr[gpuIdx][streamIdx]);

                    SAFE_FREE_HOST(svRealCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize]);
                    SAFE_FREE_HOST(svImagCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize]);

                    cudaError_t errorRealCpuMalloc = cudaMallocHost((void **)&(svRealCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize]), (svRealCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]));
                    cudaError_t errorImagCpuMalloc = cudaMallocHost((void **)&(svImagCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize]), (svImagCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]));
                    if (errorRealCpuMalloc != cudaSuccess)
                    {
                        chunkLocationRealArr[globalIndex.to_ulong() >> chunkSize] = 1;
                        std::string filePathStrReal = ssdPath + std::to_string(globalIndex.to_ulong() >> chunkSize) + ".real";
                        cufileWrite(filePathStrReal.c_str(), deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]);
                    }
                    else
                    {
                        chunkLocationRealArr[globalIndex.to_ulong() >> chunkSize] = 0;
                        cudaMemcpyAsync(svRealCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize], deviceRealCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svRealCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize], cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);
                    }

                    if (errorImagCpuMalloc != cudaSuccess)
                    {
                        chunkLocationImagArr[globalIndex.to_ulong() >> chunkSize] = 1;
                        std::string filePathStrImag = ssdPath + std::to_string(globalIndex.to_ulong() >> chunkSize) + ".imag";
                        cufileWrite(filePathStrImag.c_str(), deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize]);
                    }
                    else
                    {
                        chunkLocationImagArr[globalIndex.to_ulong() >> chunkSize] = 0;
                        cudaMemcpyAsync(svImagCompressedPtrArrHost[globalIndex.to_ulong() >> chunkSize], deviceImagCompressedBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), svImagCompressedSizePtrArrHost[globalIndex.to_ulong() >> chunkSize], cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);
                    }

                    cudaMemcpyAsync(svRealSignPtrHost[globalIndex.to_ulong() >> chunkSize], deviceRealSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), sizeof(int8_t) * exp2Int(chunkSize), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);
                    cudaMemcpyAsync(svImagSignPtrHost[globalIndex.to_ulong() >> chunkSize], deviceImagSignBufferArr[gpuIdx][streamIdx] + tmpGlobalInnerIndex * exp2Int(chunkSize), sizeof(int8_t) * exp2Int(chunkSize), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);

                    increment(globalIndex, globalInnerArr, globalInnerArrSize);
                }
#else
                cudaSetDevice(gpuIdx);
                for (IdxType tmpGlobalInnerIndex = 0; tmpGlobalInnerIndex < exp2Int(globalInnerArrSize); tmpGlobalInnerIndex++)
                {
                    cudaMemcpyAsync(svRealPtrHost + globalIndex.to_ulong(), svRealPtrArrDevice[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), sizeof(ValType) * exp2Int(chunkSize), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);
                    cudaMemcpyAsync(svImagPtrHost + globalIndex.to_ulong(), svImagPtrArrDeivce[gpuIdx] + streamIdx * (1 << (chunkSize + globalInnerArrSize)) + tmpGlobalInnerIndex * exp2Int(chunkSize), sizeof(ValType) * exp2Int(chunkSize), cudaMemcpyDeviceToHost, streamArr[gpuIdx][streamIdx]);

                    increment(globalIndex, globalInnerArr, globalInnerArrSize);
                }
#endif

                helperIdx = (helperIdx + 1) % (streamNumPerGpu * numGpus);
                gpuIdx = helperIdx / streamNumPerGpu;
                streamIdx = (streamIdx + 1) % streamNumPerGpu;
                increment(globalIndex, globalOuterArr, globalOuterArrSize);
            }

            // global synchronize
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                cudaSetDevice(tmpIdx);
                cudaDeviceSynchronize();
            }

            auto e2eEnd = std::chrono::high_resolution_clock::now();

            // calculate the compression ratio
            size_t compressedSize = thrust::reduce(thrust::host, svRealCompressedSizePtrArrHost, svRealCompressedSizePtrArrHost + chunkNum, size_t(0)) + thrust::reduce(svImagCompressedSizePtrArrHost, svImagCompressedSizePtrArrHost + chunkNum, size_t(0));
            size_t originalSize = exp2Int(numQubits) * sizeof(ValType) * 2;
            double compressionRatio = double(originalSize) / double(compressedSize);
            std::cout << "compression ratio for this stage: " << compressionRatio << std::endl;

            IdxType ssdChunkNumReal = 0;
            IdxType ssdChunkNumImag = 0;
            ssdChunkNumReal = thrust::reduce(thrust::host, chunkLocationRealArr, chunkLocationRealArr + chunkNum, 0);
            ssdChunkNumImag = thrust::reduce(thrust::host, chunkLocationImagArr, chunkLocationImagArr + chunkNum, 0);
            if (ssdChunkNumReal > 0 || ssdChunkNumImag > 0)
            {
                std::cout << "ssd backup activated" << std::endl;
                std::cout << "ssd chunk number for real part: " << ssdChunkNumReal << std::endl;
                std::cout << "ssd chunk number for imag part: " << ssdChunkNumImag << std::endl;
            }

            // Free CUDA streams
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                cudaStreamDestroy(streamArr[tmpIdx][0]);
                cudaStreamDestroy(streamArr[tmpIdx][1]);
                free(streamArr[tmpIdx]);
            }
            free(streamArr);

            // free the bitcomp plan
            for (IdxType tmpIdx = 0; tmpIdx < numGpus; tmpIdx++)
            {
                for (IdxType tmpIdx2 = 0; tmpIdx2 < 2; tmpIdx2++)
                {
                    bitcompDestroyPlan(planArr[tmpIdx][tmpIdx2]);
                }
                free(planArr[tmpIdx]);
            }
            free(planArr);

            // Calculate and print the elapsed time
            std::chrono::duration<double> elapsed = e2eEnd - e2eStart;
            std::cout << "E2E time for current stage: " << elapsed.count() << " s\n";
            // std::cout << "compression ratio for this stage: " << compressionRatio << std::endl;
            std::cout << "=================================================================== " << std::endl;
        }

        void circuitPartition()
        {
            std::set<IdxType> s = {};
            std::set<IdxType> prevS = {};
            IdxType left = 0;
            IdxType right = 0;
            IdxType prevSetSize = 0;
            IdxType stageIdx = 0;

#ifdef CIRCUIT_PARTITION_PRINT

            printf("----------------circuit partition----------------\n");

#endif

            while (right < n_gates)
            {
                while (right < n_gates)
                {
                    prevSetSize = s.size();
                    if (circuit[right]->qb0 >= chunkSize)
                    {
                        s.insert(circuit[right]->qb0);
                    }
                    if (circuit[right]->qb1 >= chunkSize)
                    {
                        s.insert(circuit[right]->qb1);
                    }
                    if (circuit[right]->qb2 >= chunkSize)
                    {
                        s.insert(circuit[right]->qb2);
                    }
                    if (circuit[right]->qb3 >= chunkSize)
                    {
                        s.insert(circuit[right]->qb3);
                    }
                    if (circuit[right]->qb4 >= chunkSize)
                    {
                        s.insert(circuit[right]->qb4);
                    }
                    if (s.size() > maxGlobalInnerNum)
                    {
                        break;
                    }
                    right++;
                    prevS = s;
                }

                if (s.size() > maxGlobalInnerNum)
                {
                    globalInnerArrSize = prevSetSize;
                }
                else
                {
                    globalInnerArrSize = s.size();
                }

                globalOuterArrSize = numQubits - chunkSize - globalInnerArrSize;

                globalOuterArr = (IdxType *)malloc(sizeof(IdxType) * globalOuterArrSize);
                globalInnerArr = (IdxType *)malloc(sizeof(IdxType) * globalInnerArrSize);

                IdxType tmpglobalInnerArrIdx = 0;
                if (s.size() <= maxGlobalInnerNum)
                {
                    for (IdxType val : s)
                    {
                        globalInnerArr[tmpglobalInnerArrIdx] = val;
                        tmpglobalInnerArrIdx++;
                    }
                }
                else
                {
                    for (IdxType val : prevS)
                    {
                        globalInnerArr[tmpglobalInnerArrIdx] = val;
                        tmpglobalInnerArrIdx++;
                    }
                }

                IdxType tmpglobalOuterArrIdx = 0;
                for (IdxType tmpIdx = chunkSize; tmpIdx < numQubits; tmpIdx++)
                {
                    if (!isInArray(globalInnerArr, globalInnerArrSize, tmpIdx))
                    {
                        globalOuterArr[tmpglobalOuterArrIdx] = tmpIdx;
                        tmpglobalOuterArrIdx++;
                    }
                }

                sort(globalOuterArr, globalOuterArr + globalOuterArrSize);
                sort(globalInnerArr, globalInnerArr + globalInnerArrSize);

#ifdef CIRCUIT_PARTITION_PRINT

                std::cout << dump(left, right);
                printf("------------------------------------------------\n");

#endif
                upload(left, right);
                getProperties();

                memq(left, right, stageIdx);

                cleanUploaded(left, right);
                clearIdx();
                s.clear();

                left = right;
                prevSetSize = 0;
                stageIdx++;
            }
        }

        void decompSv(std::string outputPath)
        {
            bitcompHandle_t plan;
            bitcompCreatePlan(
                &plan,                                // Bitcomp handle
                sizeof(ValType) * exp2Int(chunkSize), // Size in bytes of the uncompressed data
                BITCOMP_FP64_DATA,                    // Data type
                BITCOMP_LOSSY_FP_TO_SIGNED,           // Compression type
                BITCOMP_DEFAULT_ALGO);                // Bitcomp algo, default or sparse

            SAFE_ALOC_HOST(svRealPtrHost, svSize);
            SAFE_ALOC_HOST(svImagPtrHost, svSize);

            memset(svRealPtrHost, 0, svSize);
            memset(svImagPtrHost, 0, svSize);

            for (int i = 0; i < exp2Int(numQubits - chunkSize); i++)
            {
                bitcompHostUncompress(plan, svRealCompressedPtrArrHost[i], svRealPtrHost + exp2Int(chunkSize) * i);
                bitcompHostUncompress(plan, svImagCompressedPtrArrHost[i], svImagPtrHost + exp2Int(chunkSize) * i);
            }

            for (int i = 0; i < chunkNum; i++)
            {
                for (int j = 0; j < exp2Int(chunkSize); j++)
                {
                    if (svRealSignPtrHost[i][j] == 1)
                    {
                        svRealPtrHost[i * exp2Int(chunkSize) + j] = -exp2(svRealPtrHost[i * exp2Int(chunkSize) + j]);
                    }
                    else if (svRealSignPtrHost[i][j] == 0)
                    {
                        svRealPtrHost[i * exp2Int(chunkSize) + j] = exp2(svRealPtrHost[i * exp2Int(chunkSize) + j]);
                    }
                    else if (svRealSignPtrHost[i][j] == 2)
                    {
                        svRealPtrHost[i * exp2Int(chunkSize) + j] = 0;
                    }

                    if (svImagSignPtrHost[i][j] == 1)
                    {
                        svImagPtrHost[i * exp2Int(chunkSize) + j] = -exp2(svImagPtrHost[i * exp2Int(chunkSize) + j]);
                    }
                    else if (svImagSignPtrHost[i][j] == 0)
                    {
                        svImagPtrHost[i * exp2Int(chunkSize) + j] = exp2(svImagPtrHost[i * exp2Int(chunkSize) + j]);
                    }
                    else if (svImagSignPtrHost[i][j] == 2)
                    {
                        svImagPtrHost[i * exp2Int(chunkSize) + j] = 0;
                    }
                }
            }

            write_array_to_binary<double>(outputPath + "_real.bin", svRealPtrHost, exp2Int(numQubits));
            write_array_to_binary<double>(outputPath + "_imag.bin", svImagPtrHost, exp2Int(numQubits));

            SAFE_FREE_HOST(svRealPtrHost);
            SAFE_FREE_HOST(svImagPtrHost);
        }

        void beginSimulation()
        {
            getMaxChunkNum();
            deviceBufferInit();

            auto start = std::chrono::high_resolution_clock::now();
            circuitPartition();
            auto stop = std::chrono::high_resolution_clock::now();

            // Calculate duration
            auto duration = std::chrono::duration<double, std::milli>(stop - start).count();

            // Output the duration
            std::cout << "Overall end to end time: " << duration << " milliseconds" << std::endl;

            deviceBufferFree();
#ifdef ENABLE_COMPRESSION
            // decompSv();
#endif
        }

        void print_res_sv()
        {
            printf("----- Real SV ------\n");
            for (IdxType i = 0; i < dim; i++)
                printf("%lf ", svRealPtrHost[i * dim + i]);
            printf("\n");
            printf("----- Imag SV ------\n");
            for (IdxType i = 0; i < dim; i++)
                printf("%lf ", svImagPtrHost[i * dim + i]);
            printf("\n");
        }

        // =============================== Standard Gates ===================================
        // 3-parameter 2-pulse single qubit gate
        static Gate *U3(ValType theta, ValType phi, ValType lambda, IdxType m)
        {
            return new Gate(OP::U3, m, 0, 0, 0, 0, theta, phi, lambda);
        }
        // 2-parameter 1-pulse single qubit gate
        static Gate *U2(ValType phi, ValType lambda, IdxType m)
        {
            return new Gate(OP::U2, m, 0, 0, 0, 0, 0., phi, lambda);
        }
        // 1-parameter 0-pulse single qubit gate
        static Gate *U1(ValType lambda, IdxType m)
        {
            return new Gate(OP::U1, m, 0, 0, 0, 0, 0., 0., lambda);
        }
        // controlled-NOT
        static Gate *CX(IdxType m, IdxType n)
        {
            return new Gate(OP::CX, m, n, 0, 0, 0, 0., 0., 0.);
        }
        // idle gate(identity)
        static Gate *ID(IdxType m)
        {
            return new Gate(OP::ID, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Pauli gate: bit-flip
        static Gate *X(IdxType m)
        {
            return new Gate(OP::X, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Pauli gate: bit and phase flip
        static Gate *Y(IdxType m)
        {
            return new Gate(OP::Y, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Pauli gate: phase flip
        static Gate *Z(IdxType m)
        {
            return new Gate(OP::Z, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Clifford gate: Hadamard
        static Gate *H(IdxType m)
        {
            return new Gate(OP::H, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Clifford gate: sqrt(Z) phase gate
        static Gate *S(IdxType m)
        {
            return new Gate(OP::S, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Clifford gate: conjugate of sqrt(Z)
        static Gate *SDG(IdxType m)
        {
            return new Gate(OP::SDG, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // C3 gate: sqrt(S) phase gate
        static Gate *T(IdxType m)
        {
            return new Gate(OP::T, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // C3 gate: conjugate of sqrt(S)
        static Gate *TDG(IdxType m)
        {
            return new Gate(OP::TDG, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // Rotation around X-axis
        static Gate *RX(ValType theta, IdxType m)
        {
            return new Gate(OP::RX, m, 0, 0, 0, 0, theta, 0., 0.);
        }
        // Rotation around Y-axis
        static Gate *RY(ValType theta, IdxType m)
        {
            return new Gate(OP::RY, m, 0, 0, 0, 0, theta, 0., 0.);
        }
        // Rotation around Z-axis
        static Gate *RZ(ValType phi, IdxType m)
        {
            return new Gate(OP::RZ, m, 0, 0, 0, 0, 0., phi, 0.);
        }
        // =============================== Composition Gates ===================================
        // Controlled-Phase
        static Gate *CZ(IdxType m, IdxType n)
        {
            return new Gate(OP::CZ, m, n, 0, 0, 0, 0., 0., 0.);
        }
        // Controlled-Y
        static Gate *CY(IdxType m, IdxType n)
        {
            return new Gate(OP::CY, m, n, 0, 0, 0, 0., 0., 0.);
        }
        // Swap
        static Gate *SWAP(IdxType m, IdxType n)
        {
            return new Gate(OP::SWAP, m, n, 0, 0, 0, 0., 0., 0.);
        }
        // Controlled-H
        static Gate *CH(IdxType m, IdxType n)
        {
            return new Gate(OP::CH, m, n, 0, 0, 0, 0., 0., 0.);
        }
        // C3 gate: Toffoli
        static Gate *CCX(IdxType l, IdxType m, IdxType n)
        {
            return new Gate(OP::CCX, l, m, n, 0, 0, 0., 0., 0.);
        }
        // Fredkin gate
        static Gate *CSWAP(IdxType l, IdxType m, IdxType n)
        {
            return new Gate(OP::CSWAP, l, m, n, 0, 0, 0., 0., 0.);
        }
        // Controlled RX rotation
        static Gate *CRX(ValType lambda, IdxType m, IdxType n)
        {
            return new Gate(OP::CRX, m, n, 0, 0, 0, 0., 0., lambda);
        }
        // Controlled RY rotation
        static Gate *CRY(ValType lambda, IdxType m, IdxType n)
        {
            return new Gate(OP::CRY, m, n, 0, 0, 0, 0., 0., lambda);
        }
        // Controlled RZ rotation
        static Gate *CRZ(ValType lambda, IdxType m, IdxType n)
        {
            return new Gate(OP::CRZ, m, n, 0, 0, 0, 0., 0., lambda);
        }
        // Controlled phase rotation
        static Gate *CU1(ValType lambda, IdxType m, IdxType n)
        {
            return new Gate(OP::CU1, m, n, 0, 0, 0, 0., 0., lambda);
        }
        // Controlled-U
        static Gate *CU3(ValType theta, ValType phi, ValType lambda, IdxType m, IdxType n)
        {
            return new Gate(OP::CU3, m, n, 0, 0, 0, theta, phi, lambda);
        }
        // 2-qubit XX rotation
        static Gate *RXX(ValType theta, IdxType m, IdxType n)
        {
            return new Gate(OP::RXX, m, n, 0, 0, 0, theta, 0., 0.);
        }
        // 2-qubit ZZ rotation
        static Gate *RZZ(ValType theta, IdxType m, IdxType n)
        {
            return new Gate(OP::RZZ, m, n, 0, 0, 0, theta, 0., 0.);
        }
        // Relative-phase CCX
        static Gate *RCCX(IdxType l, IdxType m, IdxType n)
        {
            return new Gate(OP::RCCX, l, m, n, 0, 0, 0., 0., 0.);
        }
        // Relative-phase 3-controlled X gate
        static Gate *RC3X(IdxType l, IdxType m, IdxType n, IdxType o)
        {
            return new Gate(OP::RC3X, l, m, n, o, 0, 0., 0., 0.);
        }
        // 3-controlled X gate
        static Gate *C3X(IdxType l, IdxType m, IdxType n, IdxType o)
        {
            return new Gate(OP::C3X, l, m, n, o, 0, 0., 0., 0.);
        }
        // 3-controlled sqrt(X) gate
        static Gate *C3SQRTX(IdxType l, IdxType m, IdxType n, IdxType o)
        {
            return new Gate(OP::C3SQRTX, l, m, n, o, 0, 0., 0., 0.);
        }
        // 4-controlled X gate
        static Gate *C4X(IdxType l, IdxType m, IdxType n, IdxType o, IdxType p)
        {
            return new Gate(OP::C4X, l, m, n, o, p, 0., 0., 0.);
        }
        // =============================== sv_Sim Native Gates ===================================
        static Gate *R(ValType theta, IdxType m)
        {
            return new Gate(OP::R, m, 0, 0, 0, 0, theta, 0., 0.);
        }
        static Gate *SRN(IdxType m)
        {
            return new Gate(OP::SRN, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        static Gate *W(IdxType m)
        {
            return new Gate(OP::W, m, 0, 0, 0, 0, 0., 0., 0.);
        }
        // 2-qubit YY rotation
        static Gate *RYY(ValType theta, IdxType m, IdxType n)
        {
            return new Gate(OP::RYY, m, n, 0, 0, 0, theta, 0., 0.);
        }

    public:
        // numQubits is the number of qubits
        const IdxType numQubits;
        // gpu_scale is 2^x of the number of GPUs, e.g., with 8 GPUs the gpu_scale is 3 (2^3=8)
        const IdxType gpu_scale;
        const IdxType numGpus;
        const IdxType dim;
        const IdxType half_dim;
        const IdxType lg2_m_gpu;
        const IdxType m_gpu;

        const IdxType svSize;
        const IdxType svSizePerGpu;

        IdxType n_gates;
        // CPU arrays
        ValType *svRealPtrHost;
        ValType *svImagPtrHost;

        int8_t **svRealSignPtrHost;
        int8_t **svImagSignPtrHost;

        // GPU pointers on CPU
        ValType **svRealPtrArrDevice;
        ValType **svImagPtrArrDeivce;

        ValType gpu_mem;
        // hold the CPU-side gates
        vector<Gate *> circuit;
        // for freeing GPU-side gates in clear(), otherwise there can be GPU memory leak
        vector<Gate *> *circuitCopy;
        // hold the GPU-side gates
        Gate **circuit_gpu;
        // hold the GPU-side simulator instances
        Simulation **sim_gpu;

        IdxType chunkSize;
        IdxType chunkNum;

        ValType **svRealCompressedPtrArrHost;
        ValType **svImagCompressedPtrArrHost;

        size_t *svRealCompressedSizePtrArrHost;
        size_t *svImagCompressedSizePtrArrHost;

        ValType **svRealBufferPtrArr;
        ValType **svImagBufferPtrArr;

        ValType **deviceSvRealBufferPtrArr;
        ValType **deviceSvImagBufferPtrArr;

        // IdxType outerSize;
        // IdxType innerSize;

        // IdxType *outerArr;
        // IdxType *innerArr;

        IdxType globalOuterArrSize;
        IdxType globalInnerArrSize;

        IdxType *globalOuterArr;
        IdxType *globalInnerArr;

        IdxType maxGlobalInnerNum;

        ValType ***deviceRealCompressedBufferArr;
        ValType ***deviceImagCompressedBufferArr;

        int8_t ***deviceRealSignBufferArr;
        int8_t ***deviceImagSignBufferArr;

        size_t **compressedSizeRealDevice;
        size_t **compressedSizeImagDevice;

        IdxType streamNumPerGpu;
        IdxType streamNumPerGpuLog2;

        int8_t *chunkLocationRealArr;
        int8_t *chunkLocationImagArr;

        std::string ssdPath;

        int *numBlocksPerSm;

        cudaDeviceProp *deviceProp;
    };

    __global__ void simulation_kernel(Simulation *sim, IdxType gpuIdx, IdxType start, IdxType end, IdxType simulationOffset)
    {
        grid_group grid = this_grid();

        for (IdxType t = 0; t < end - start; t++)
        {
            ((sim->circuit_gpu)[t])->exe_op(sim, sim->svRealPtrArrDevice, sim->svImagPtrArrDeivce, gpuIdx, simulationOffset);
        }
    }

//=================================== Gate Definition ==========================================

// Define MG-BSP machine operation header (Original version with semantics)
 //  #define OP_HEAD_ORIGIN grid_group grid = this_grid(); \
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; \
    const IdxType outer_bound = (1 << ( (sim->numQubits) - qubit - 1)); \
    const IdxType inner_bound = (1 << qubit); \
        for (IdxType i = tid;i<outer_bound*inner_bound;\
                i+=blockDim.x*gridDim.x){ \
            IdxType outer = i / inner_bound; \
            IdxType inner =  i % inner_bound; \
            IdxType offset = (2 * outer) * inner_bound; \
            IdxType pos0 = offset + inner; \
            IdxType pos1 = pos0 + inner_bound;

// Define MG-BSP machine operation header (Optimized version)
#define OP_HEAD                                                                   \
    grid_group grid = this_grid();                                                \
    for (IdxType i = grid.thread_rank(); i < (sim->half_dim);                     \
         i += grid.size())                                                        \
    {                                                                             \
        IdxType outer = (i >> qubit);                                             \
        IdxType inner = (i & ((1 << qubit) - 1));                                 \
        IdxType offset = (outer << (qubit + 1));                                  \
        IdxType pos0_gid = ((offset + inner) >> (sim->lg2_m_gpu));                \
        IdxType pos0 = ((offset + inner) & (sim->m_gpu - 1));                     \
        IdxType pos1_gid = ((offset + inner + (1 << qubit)) >> (sim->lg2_m_gpu)); \
        IdxType pos1 = ((offset + inner + (1 << qubit)) & (sim->m_gpu - 1));

    // IdxType pos0_gid = ((offset + inner) >> (sim->lg2_m_gpu));                \
    // IdxType pos1_gid = ((offset + inner + (1 << qubit)) >> (sim->lg2_m_gpu)); \

    /* Muti-GPUs equally share the sv_real and sv_imag, we need
       to figure out which GPU the target address sits (pos_gid) and what
       is the id in that segment (pos) */

// Define MG-BSP machine operation footer
#define OP_TAIL \
    }           \
    grid.sync();

#define nOP_HEAD                                                                                                \
    grid_group grid = this_grid();                                                                              \
    for (IdxType i = grid.thread_rank(); i < (1 << (sim->chunkSize + sim->globalInnerArrSize - 1));             \
         i += grid.size())                                                                                      \
    {                                                                                                           \
        IdxType outer = (i >> qubit);                                                                           \
        IdxType inner = (i & ((1 << qubit) - 1));                                                               \
        IdxType offset = (outer << (qubit + 1));                                                                \
        IdxType pos0_gid = gpuIdx;                                                                              \
        IdxType pos0 = (offset + inner) + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize)); \
        IdxType pos1_gid = gpuIdx;                                                                              \
        IdxType pos1 = (offset + inner + (1 << qubit)) + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));

#define nOP_TAIL \
    }            \
    grid.sync();

    //============== Unified 1-qubit Gate ================
    __device__ __inline__ void C1_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType e0_real, const ValType e0_imag,
                                       const ValType e1_real, const ValType e1_imag,
                                       const ValType e2_real, const ValType e2_imag,
                                       const ValType e3_real, const ValType e3_imag,
                                       const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = (e0_real * el0_real) - (e0_imag * el0_imag) + (e1_real * el1_real) - (e1_imag * el1_imag);
        svImagPtrArrDeivce[pos0_gid][pos0] = (e0_real * el0_imag) + (e0_imag * el0_real) + (e1_real * el1_imag) + (e1_imag * el1_real);
        svRealPtrArrDevice[pos1_gid][pos1] = (e2_real * el0_real) - (e2_imag * el0_imag) + (e3_real * el1_real) - (e3_imag * el1_imag);
        svImagPtrArrDeivce[pos1_gid][pos1] = (e2_real * el0_imag) + (e2_imag * el0_real) + (e3_real * el1_imag) + (e3_imag * el1_real);
        nOP_TAIL;
    }

    //============== Unified 2-qubit Gate ================
    __device__ __inline__ void C2_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType e00_real, const ValType e00_imag,
                                       const ValType e01_real, const ValType e01_imag,
                                       const ValType e02_real, const ValType e02_imag,
                                       const ValType e03_real, const ValType e03_imag,
                                       const ValType e10_real, const ValType e10_imag,
                                       const ValType e11_real, const ValType e11_imag,
                                       const ValType e12_real, const ValType e12_imag,
                                       const ValType e13_real, const ValType e13_imag,
                                       const ValType e20_real, const ValType e20_imag,
                                       const ValType e21_real, const ValType e21_imag,
                                       const ValType e22_real, const ValType e22_imag,
                                       const ValType e23_real, const ValType e23_imag,
                                       const ValType e30_real, const ValType e30_imag,
                                       const ValType e31_real, const ValType e31_imag,
                                       const ValType e32_real, const ValType e32_imag,
                                       const ValType e33_real, const ValType e33_imag,
                                       const IdxType qubit1, const IdxType qubit2, IdxType gpuIdx, IdxType simulationOffset)
    {
        grid_group grid = this_grid();
        const IdxType q0dim = (1 << max(qubit1, qubit2));
        const IdxType q1dim = (1 << min(qubit1, qubit2));
        assert(qubit1 != qubit2); // Non-cloning
        const IdxType outer_factor = ((1 << (sim->chunkSize + sim->globalInnerArrSize)) + q0dim + q0dim - 1) >> (max(qubit1, qubit2) + 1);
        const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(qubit1, qubit2) + 1);
        const IdxType inner_factor = q1dim;
        const IdxType qubit1_dim = (1 << qubit1);
        const IdxType qubit2_dim = (1 << qubit2);

        for (IdxType i = grid.thread_rank(); i < (1 << (sim->chunkSize + sim->globalInnerArrSize - 2));
             i += grid.size())
        {
            IdxType outer = ((i / inner_factor) / (mider_factor)) * (q0dim + q0dim);
            IdxType mider = ((i / inner_factor) % (mider_factor)) * (q1dim + q1dim);
            IdxType inner = i % inner_factor;
            IdxType pos0_org = outer + mider + inner;
            IdxType pos1_org = outer + mider + inner + qubit2_dim;
            IdxType pos2_org = outer + mider + inner + qubit1_dim;
            IdxType pos3_org = outer + mider + inner + q0dim + q1dim;

            IdxType pos0_gid = gpuIdx;
            IdxType pos1_gid = gpuIdx;
            IdxType pos2_gid = gpuIdx;
            IdxType pos3_gid = gpuIdx;

            IdxType pos0 = pos0_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));
            IdxType pos1 = pos1_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));
            IdxType pos2 = pos2_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));
            IdxType pos3 = pos3_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));

            const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
            const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
            const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
            const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
            const ValType el2_real = svRealPtrArrDevice[pos2_gid][pos2];
            const ValType el2_imag = svImagPtrArrDeivce[pos2_gid][pos2];
            const ValType el3_real = svRealPtrArrDevice[pos3_gid][pos3];
            const ValType el3_imag = svImagPtrArrDeivce[pos3_gid][pos3];

            // Real part
            svRealPtrArrDevice[pos0_gid][pos0] = (e00_real * el0_real) - (e00_imag * el0_imag) + (e01_real * el1_real) - (e01_imag * el1_imag) + (e02_real * el2_real) - (e02_imag * el2_imag) + (e03_real * el3_real) - (e03_imag * el3_imag);
            svRealPtrArrDevice[pos1_gid][pos1] = (e10_real * el0_real) - (e10_imag * el0_imag) + (e11_real * el1_real) - (e11_imag * el1_imag) + (e12_real * el2_real) - (e12_imag * el2_imag) + (e13_real * el3_real) - (e13_imag * el3_imag);
            svRealPtrArrDevice[pos2_gid][pos2] = (e20_real * el0_real) - (e20_imag * el0_imag) + (e21_real * el1_real) - (e21_imag * el1_imag) + (e22_real * el2_real) - (e22_imag * el2_imag) + (e23_real * el3_real) - (e23_imag * el3_imag);
            svRealPtrArrDevice[pos3_gid][pos3] = (e30_real * el0_real) - (e30_imag * el0_imag) + (e31_real * el1_real) - (e31_imag * el1_imag) + (e32_real * el2_real) - (e32_imag * el2_imag) + (e33_real * el3_real) - (e33_imag * el3_imag);

            // Imag part
            svImagPtrArrDeivce[pos0_gid][pos0] = (e00_real * el0_imag) + (e00_imag * el0_real) + (e01_real * el1_imag) + (e01_imag * el1_real) + (e02_real * el2_imag) + (e02_imag * el2_real) + (e03_real * el3_imag) + (e03_imag * el3_real);
            svImagPtrArrDeivce[pos1_gid][pos1] = (e10_real * el0_imag) + (e10_imag * el0_real) + (e11_real * el1_imag) + (e11_imag * el1_real) + (e12_real * el2_imag) + (e12_imag * el2_real) + (e13_real * el3_imag) + (e13_imag * el3_real);
            svImagPtrArrDeivce[pos2_gid][pos2] = (e20_real * el0_imag) + (e20_imag * el0_real) + (e21_real * el1_imag) + (e21_imag * el1_real) + (e22_real * el2_imag) + (e22_imag * el2_real) + (e23_real * el3_imag) + (e23_imag * el3_real);
            svImagPtrArrDeivce[pos3_gid][pos3] = (e30_real * el0_imag) + (e30_imag * el0_real) + (e31_real * el1_imag) + (e31_imag * el1_real) + (e32_real * el2_imag) + (e32_imag * el2_real) + (e33_real * el3_imag) + (e33_imag * el3_real);
        }
        grid.sync();
    }

    //============== CX Gate ================
    // Controlled-NOT or CNOT
    /** CX   = [1 0 0 0]
               [0 1 0 0]
               [0 0 0 1]
               [0 0 1 0]
    */
    __device__ __inline__ void CX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const IdxType ctrl, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        grid_group grid = this_grid();
        const IdxType q0dim = (1 << max(ctrl, qubit));
        const IdxType q1dim = (1 << min(ctrl, qubit));
        // assert(ctrl != qubit); // Non-cloning
        const IdxType outer_factor = ((1 << (sim->chunkSize + sim->globalInnerArrSize)) + q0dim + q0dim - 1) >> (max(ctrl, qubit) + 1);
        const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl, qubit) + 1);
        const IdxType inner_factor = q1dim;
        const IdxType ctrldim = (1 << ctrl);

        for (IdxType i = grid.thread_rank(); i < (1 << (sim->chunkSize + sim->globalInnerArrSize - 2));
             i += grid.size())
        {
            IdxType outer = ((i / inner_factor) / (mider_factor)) * (q0dim + q0dim);
            IdxType mider = ((i / inner_factor) % (mider_factor)) * (q1dim + q1dim);
            IdxType inner = i % inner_factor;

            IdxType pos0_org = outer + mider + inner + ctrldim;
            IdxType pos1_org = outer + mider + inner + q0dim + q1dim;
            IdxType pos0_gid = gpuIdx;
            IdxType pos1_gid = gpuIdx;
            IdxType pos0 = pos0_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));
            IdxType pos1 = pos1_org + simulationOffset * (1 << (sim->chunkSize + sim->globalInnerArrSize));
            const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
            const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
            const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
            const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
            svRealPtrArrDevice[pos0_gid][pos0] = el1_real;
            svImagPtrArrDeivce[pos0_gid][pos0] = el1_imag;
            svRealPtrArrDevice[pos1_gid][pos1] = el0_real;
            svImagPtrArrDeivce[pos1_gid][pos1] = el0_imag;
        }
        grid.sync();
    }

    //============== X Gate ================
    // Pauli gate: bit flip
    /** X = [0 1]
            [1 0]
    */
    __device__ __inline__ void X_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = el1_real;
        svImagPtrArrDeivce[pos0_gid][pos0] = el1_imag;
        svRealPtrArrDevice[pos1_gid][pos1] = el0_real;
        svImagPtrArrDeivce[pos1_gid][pos1] = el0_imag;
        nOP_TAIL;
    }

    //============== Y Gate ================
    // Pauli gate: bit and phase flip
    /** Y = [0 -i]
            [i  0]
    */
    __device__ __inline__ void Y_GATE(const Simulation *sim, ValType **svRealPtrArrDevice,
                                      ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = el1_imag;
        svImagPtrArrDeivce[pos0_gid][pos0] = -el1_real;
        svRealPtrArrDevice[pos1_gid][pos1] = -el0_imag;
        svImagPtrArrDeivce[pos1_gid][pos1] = el0_real;
        nOP_TAIL;
    }

    //============== Z Gate ================
    // Pauli gate: phase flip
    /** Z = [1  0]
            [0 -1]
    */
    __device__ __inline__ void Z_GATE(const Simulation *sim, ValType **svRealPtrArrDevice,
                                      ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = -el1_real;
        svImagPtrArrDeivce[pos1_gid][pos1] = -el1_imag;
        nOP_TAIL;
    }

    //============== H Gate ================
    // Clifford gate: Hadamard
    /** H = 1/sqrt(2) * [1  1]
                        [1 -1]
    */
    __device__ __inline__ void H_GATE(const Simulation *sim, ValType **svRealPtrArrDevice,
                                      ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = S2I * (el0_real + el1_real);
        svImagPtrArrDeivce[pos0_gid][pos0] = S2I * (el0_imag + el1_imag);
        svRealPtrArrDevice[pos1_gid][pos1] = S2I * (el0_real - el1_real);
        svImagPtrArrDeivce[pos1_gid][pos1] = S2I * (el0_imag - el1_imag);
        nOP_TAIL;
    }

    //============== SRN Gate ================
    // Square Root of X gate, it maps |0> to ((1+i)|0>+(1-i)|1>)/2,
    // and |1> to ((1-i)|0>+(1+i)|1>)/2
    /** SRN = 1/2 * [1+i 1-i]
                    [1-i 1+1]
    */
    __device__ __inline__ void SRN_GATE(const Simulation *sim, ValType **svRealPtrArrDevice,
                                        ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = 0.5 * (el0_real + el1_real);
        svImagPtrArrDeivce[pos0_gid][pos0] = 0.5 * (el0_imag - el1_imag);
        svRealPtrArrDevice[pos1_gid][pos1] = 0.5 * (el0_real + el1_real);
        svImagPtrArrDeivce[pos1_gid][pos1] = 0.5 * (-el0_imag + el1_imag);
        nOP_TAIL;
    }

    //============== ID Gate ================
    /** ID = [1 0]
             [0 1]
    */
    __device__ __inline__ void ID_GATE(const Simulation *sim, ValType **svRealPtrArrDevice,
                                       ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
    }

    //============== R Gate ================
    // Phase-shift gate, it leaves |0> unchanged
    // and maps |1> to e^{i\psi}|1>
    /** R = [1 0]
            [0 0+p*i]
    */
    __device__ __inline__ void R_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                      const ValType phase, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = -(el1_imag * phase);
        svImagPtrArrDeivce[pos1_gid][pos1] = el1_real * phase;
        nOP_TAIL;
    }

    //============== S Gate ================
    // Clifford gate: sqrt(Z) phase gate
    /** S = [1 0]
            [0 i]
    */
    __device__ __inline__ void S_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = -el1_imag;
        svImagPtrArrDeivce[pos1_gid][pos1] = el1_real;
        nOP_TAIL;
    }

    //============== SDG Gate ================
    // Clifford gate: conjugate of sqrt(Z) phase gate
    /** SDG = [1  0]
              [0 -i]
    */
    __device__ __inline__ void SDG_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = el1_imag;
        svImagPtrArrDeivce[pos1_gid][pos1] = -el1_real;
        nOP_TAIL;
    }

    //============== T Gate ================
    // C3 gate: sqrt(S) phase gate
    /** T = [1 0]
            [0 s2i+s2i*i]
    */
    __device__ __inline__ void T_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = S2I * (el1_real - el1_imag);
        svImagPtrArrDeivce[pos1_gid][pos1] = S2I * (el1_real + el1_imag);
        nOP_TAIL;
    }

    //============== TDG Gate ================
    // C3 gate: conjugate of sqrt(S) phase gate
    /** TDG = [1 0]
              [0 s2i-s2i*i]
    */
    __device__ __inline__ void TDG_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos1_gid][pos1] = S2I * (el1_real + el1_imag);
        svImagPtrArrDeivce[pos1_gid][pos1] = S2I * (-el1_real + el1_imag);
        nOP_TAIL;
    }

    //============== D Gate ================
    /** D = [e0_real+i*e0_imag 0]
            [0 e3_real+i*e3_imag]
    */
    __device__ __inline__ void D_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                      const ValType e0_real, const ValType e0_imag,
                                      const ValType e3_real, const ValType e3_imag,
                                      const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = (e0_real * el0_real) - (e0_imag * el0_imag);
        svImagPtrArrDeivce[pos0_gid][pos0] = (e0_real * el0_imag) + (e0_imag * el0_real);
        svRealPtrArrDevice[pos1_gid][pos1] = (e3_real * el1_real) - (e3_imag * el1_imag);
        svImagPtrArrDeivce[pos1_gid][pos1] = (e3_real * el1_imag) + (e3_imag * el1_real);
        nOP_TAIL;
    }

    //============== U1 Gate ================
    // 1-parameter 0-pulse single qubit gate
    __device__ __inline__ void U1_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType lambda, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType e3_real = cos(lambda);
        ValType e3_imag = sin(lambda);

        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = el0_real;
        svImagPtrArrDeivce[pos0_gid][pos0] = el0_imag;
        svRealPtrArrDevice[pos1_gid][pos1] = (e3_real * el1_real) - (e3_imag * el1_imag);
        svImagPtrArrDeivce[pos1_gid][pos1] = (e3_real * el1_imag) + (e3_imag * el1_real);
        nOP_TAIL;
    }

    //============== U2 Gate ================
    // 2-parameter 1-pulse single qubit gate
    __device__ __inline__ void U2_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType phi, const ValType lambda, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType e0_real = S2I;
        ValType e0_imag = 0;
        ValType e1_real = -S2I * cos(lambda);
        ValType e1_imag = -S2I * sin(lambda);
        ValType e2_real = S2I * cos(phi);
        ValType e2_imag = S2I * sin(phi);
        ValType e3_real = S2I * cos(phi + lambda);
        ValType e3_imag = S2I * sin(phi + lambda);
        C1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, e0_real, e0_imag, e1_real, e1_imag,
                e2_real, e2_imag, e3_real, e3_imag, qubit, gpuIdx, simulationOffset);
    }

    //============== U3 Gate ================
    // 3-parameter 2-pulse single qubit gate
    __device__ __inline__ void U3_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType theta, const ValType phi,
                                       const ValType lambda, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType e0_real = cos(theta / 2.);
        ValType e0_imag = 0;
        ValType e1_real = -cos(lambda) * sin(theta / 2.);
        ValType e1_imag = -sin(lambda) * sin(theta / 2.);
        ValType e2_real = cos(phi) * sin(theta / 2.);
        ValType e2_imag = sin(phi) * sin(theta / 2.);
        ValType e3_real = cos(phi + lambda) * cos(theta / 2.);
        ValType e3_imag = sin(phi + lambda) * cos(theta / 2.);
        C1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, e0_real, e0_imag, e1_real, e1_imag,
                e2_real, e2_imag, e3_real, e3_imag, qubit, gpuIdx, simulationOffset);
    }

    //============== RX Gate ================
    // Rotation around X-axis
    __device__ __inline__ void RX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType theta, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType rx_real = cos(theta / 2.0);
        ValType rx_imag = -sin(theta / 2.0);
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = (rx_real * el0_real) - (rx_imag * el1_imag);
        svImagPtrArrDeivce[pos0_gid][pos0] = (rx_real * el0_imag) + (rx_imag * el1_real);
        svRealPtrArrDevice[pos1_gid][pos1] = -(rx_imag * el0_imag) + (rx_real * el1_real);
        svImagPtrArrDeivce[pos1_gid][pos1] = +(rx_imag * el0_real) + (rx_real * el1_imag);
        nOP_TAIL;
    }

    //============== RY Gate ================
    // Rotation around Y-axis
    __device__ __inline__ void RY_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType theta, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType e0_real = cos(theta / 2.0);
        ValType e1_real = -sin(theta / 2.0);
        ValType e2_real = sin(theta / 2.0);
        ValType e3_real = cos(theta / 2.0);

        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = (e0_real * el0_real) + (e1_real * el1_real);
        svImagPtrArrDeivce[pos0_gid][pos0] = (e0_real * el0_imag) + (e1_real * el1_imag);
        svRealPtrArrDevice[pos1_gid][pos1] = (e2_real * el0_real) + (e3_real * el1_real);
        svImagPtrArrDeivce[pos1_gid][pos1] = (e2_real * el0_imag) + (e3_real * el1_imag);
        nOP_TAIL;
    }

    //============== RZ Gate ================
    // Rotation around Z-axis
    __device__ __inline__ void RZ_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const ValType phi, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, phi, qubit, gpuIdx, simulationOffset);
    }

    //============== CZ Gate ================
    // Controlled-Phase
    __device__ __inline__ void CZ_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
    }

    //============== CY Gate ================
    // Controlled-Y
    __device__ __inline__ void CY_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        SDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        S_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
    }

    //============== CH Gate ================
    // Controlled-H
    __device__ __inline__ void CH_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                       const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        SDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        S_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        S_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, gpuIdx, simulationOffset);
    }

    //============== CRZ Gate ================
    // Controlled RZ rotation
    __device__ __inline__ void CRZ_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType lambda, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, lambda / 2, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -lambda / 2, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
    }

    //============== CU1 Gate ================
    // Controlled phase rotation
    __device__ __inline__ void CU1_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType lambda, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, lambda / 2, a, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -lambda / 2, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, lambda / 2, b, gpuIdx, simulationOffset);
    }

    //============== CU3 Gate ================
    // Controlled U
    __device__ __inline__ void CU3_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType theta, const ValType phi, const ValType lambda,
                                        const IdxType c, const IdxType t, IdxType gpuIdx, IdxType simulationOffset)
    {
        ValType temp1 = (lambda - phi) / 2;
        ValType temp2 = theta / 2;
        ValType temp3 = -(phi + lambda) / 2;
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -temp3, c, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, temp1, t, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, t, gpuIdx, simulationOffset);
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -temp2, 0, temp3, t, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, t, gpuIdx, simulationOffset);
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, temp2, phi, 0, t, gpuIdx, simulationOffset);
    }

    //========= Toffoli Gate ==========
    __device__ __inline__ void CCX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const IdxType a, const IdxType b, const IdxType c, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        TDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        TDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, gpuIdx, simulationOffset);
        TDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
    }

    //========= SWAP Gate ==========
    __device__ __inline__ void SWAP_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                         const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, a, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
    }

    //========= Fredkin Gate ==========
    __device__ __inline__ void CSWAP_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                          const IdxType a, const IdxType b, const IdxType c, IdxType gpuIdx, IdxType simulationOffset)
    {
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, b, gpuIdx, simulationOffset);
        CCX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, b, gpuIdx, simulationOffset);
    }

    //============== CRX Gate ================
    // Controlled RX rotation
    __device__ __inline__ void CRX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType lambda, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 2, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -lambda / 2, 0, 0, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, lambda / 2, -PI / 2, 0, b, gpuIdx, simulationOffset);
    }

    //============== CRY Gate ================
    // Controlled RY rotation
    __device__ __inline__ void CRY_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType lambda, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, lambda / 2, 0, 0, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -lambda / 2, 0, 0, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
    }

    //============== RXX Gate ================
    // 2-qubit XX rotation
    __device__ __inline__ void RXX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType theta, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 2, theta, 0, a, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -theta, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, gpuIdx, simulationOffset);
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI, PI - theta, a, gpuIdx, simulationOffset);
    }

    //============== RZZ Gate ================
    // 2-qubit ZZ rotation
    __device__ __inline__ void RZZ_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType theta, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, theta, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
    }

    //============== RCCX Gate ================
    // Relative-phase CCX
    __device__ __inline__ void RCCX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                         const IdxType a, const IdxType b, const IdxType c, IdxType gpuIdx, IdxType simulationOffset)
    {
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, c, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, c, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, c, gpuIdx, simulationOffset);
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, c, gpuIdx, simulationOffset);
    }

    //============== RC3X Gate ================
    // Relative-phase 3-controlled X gate
    __device__ __inline__ void RC3X_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                         const IdxType a, const IdxType b, const IdxType c, const IdxType d, IdxType gpuIdx, IdxType simulationOffset)
    {
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, d, gpuIdx, simulationOffset);
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, d, gpuIdx, simulationOffset);
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, c, d, gpuIdx, simulationOffset);
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, d, gpuIdx, simulationOffset);
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, 0, PI, d, gpuIdx, simulationOffset);
    }

    //============== C3X Gate ================
    // 3-controlled X gate
    __device__ __inline__ void C3X_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const IdxType a, const IdxType b, const IdxType c, const IdxType d, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, a, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, b, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, b, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 4, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
    }

    //============== C3SQRTX Gate ================
    // 3-controlled sqrt(X) gate, this equals the C3X gate where the CU1
    // rotations are -PI/8 not -PI/4
    __device__ __inline__ void C3SQRTX_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                            const IdxType a, const IdxType b, const IdxType c, const IdxType d, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 8, a, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 8, b, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 8, b, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 8, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 8, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, b, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 8, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, c, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 8, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
    }

    //============== C4X Gate ================
    // 4-controlled X gate
    __device__ __inline__ void C4X_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const IdxType a, const IdxType b, const IdxType c,
                                        const IdxType d, const IdxType e, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, e, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 2, d, e, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, e, gpuIdx, simulationOffset);
        C3X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, c, d, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 4, d, e, gpuIdx, simulationOffset);
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, d, gpuIdx, simulationOffset);
        C3X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, c, d, gpuIdx, simulationOffset);
        C3SQRTX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, c, e, gpuIdx, simulationOffset);
    }

    //============== W Gate ================
    // W gate: e^(-i*pi/4*X)
    /** W = [s2i    -s2i*i]
            [-s2i*i s2i   ]
    */
    __device__ __inline__ void W_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, const IdxType qubit, IdxType gpuIdx, IdxType simulationOffset)
    {
        nOP_HEAD;
        const ValType el0_real = svRealPtrArrDevice[pos0_gid][pos0];
        const ValType el0_imag = svImagPtrArrDeivce[pos0_gid][pos0];
        const ValType el1_real = svRealPtrArrDevice[pos1_gid][pos1];
        const ValType el1_imag = svImagPtrArrDeivce[pos1_gid][pos1];
        svRealPtrArrDevice[pos0_gid][pos0] = S2I * (el0_real + el1_imag);
        svImagPtrArrDeivce[pos0_gid][pos0] = S2I * (el0_imag - el1_real);
        svRealPtrArrDevice[pos1_gid][pos1] = S2I * (el0_imag + el1_real);
        svImagPtrArrDeivce[pos1_gid][pos1] = S2I * (-el0_real + el1_imag);
        nOP_TAIL;
    }

    //============== RYY Gate ================
    // 2-qubit YY rotation
    __device__ __inline__ void RYY_GATE(const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce,
                                        const ValType theta, const IdxType a, const IdxType b, IdxType gpuIdx, IdxType simulationOffset)
    {
        RX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 2, a, gpuIdx, simulationOffset);
        RX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, PI / 2, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        RZ_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, theta, b, gpuIdx, simulationOffset);
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, a, b, gpuIdx, simulationOffset);
        RX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 2, a, gpuIdx, simulationOffset);
        RX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, -PI / 2, b, gpuIdx, simulationOffset);
    }

    //==================================== Gate Ops  ========================================

    __device__ void U3_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        U3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->phi, g->lambda, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void U2_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        U2_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->phi, g->lambda, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void U1_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        U1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->lambda, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void CX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void ID_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        ID_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void X_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void Y_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        Y_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void Z_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        Z_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void H_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        H_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void S_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        S_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void SDG_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        SDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void T_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        T_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void TDG_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        TDG_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void RX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void RY_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RY_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void RZ_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RZ_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->phi, g->qb0, gpuIdx, simulationOffset);
    }

    // Composition Ops
    __device__ void CZ_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CZ_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CY_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CY_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void SWAP_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        SWAP_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CH_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CH_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CCX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CCX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, gpuIdx, simulationOffset);
    }

    __device__ void CSWAP_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CSWAP_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, gpuIdx, simulationOffset);
    }

    __device__ void CRX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CRX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->lambda, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CRY_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CRY_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->lambda, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CRZ_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CRZ_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->lambda, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CU1_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CU1_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->lambda, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void CU3_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        CU3_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->phi, g->lambda, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void RXX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RXX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void RZZ_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RZZ_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    __device__ void RCCX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RCCX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, gpuIdx, simulationOffset);
    }

    __device__ void RC3X_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RC3X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, g->qb3, gpuIdx, simulationOffset);
    }

    __device__ void C3X_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        C3X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, g->qb3, gpuIdx, simulationOffset);
    }

    __device__ void C3SQRTX_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        C3SQRTX_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, g->qb3, gpuIdx, simulationOffset);
    }

    __device__ void C4X_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        C4X_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, g->qb1, g->qb2, g->qb3, g->qb4, gpuIdx, simulationOffset);
    }

    __device__ void R_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        R_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, gpuIdx, simulationOffset);
    }
    __device__ void SRN_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        SRN_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }
    __device__ void W_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        W_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->qb0, gpuIdx, simulationOffset);
    }

    __device__ void RYY_OP(const Gate *g, const Simulation *sim, ValType **svRealPtrArrDevice, ValType **svImagPtrArrDeivce, IdxType gpuIdx, IdxType simulationOffset)
    {
        RYY_GATE(sim, svRealPtrArrDevice, svImagPtrArrDeivce, g->theta, g->qb0, g->qb1, gpuIdx, simulationOffset);
    }

    // ============================ Device Function Pointers ================================
    __device__ func_t pU3_OP = U3_OP;
    __device__ func_t pU2_OP = U2_OP;
    __device__ func_t pU1_OP = U1_OP;
    __device__ func_t pCX_OP = CX_OP;
    __device__ func_t pID_OP = ID_OP;
    __device__ func_t pX_OP = X_OP;
    __device__ func_t pY_OP = Y_OP;
    __device__ func_t pZ_OP = Z_OP;
    __device__ func_t pH_OP = H_OP;
    __device__ func_t pS_OP = S_OP;
    __device__ func_t pSDG_OP = SDG_OP;
    __device__ func_t pT_OP = T_OP;
    __device__ func_t pTDG_OP = TDG_OP;
    __device__ func_t pRX_OP = RX_OP;
    __device__ func_t pRY_OP = RY_OP;
    __device__ func_t pRZ_OP = RZ_OP;
    __device__ func_t pCZ_OP = CZ_OP;
    __device__ func_t pCY_OP = CY_OP;
    __device__ func_t pSWAP_OP = SWAP_OP;
    __device__ func_t pCH_OP = CH_OP;
    __device__ func_t pCCX_OP = CCX_OP;
    __device__ func_t pCSWAP_OP = CSWAP_OP;
    __device__ func_t pCRX_OP = CRX_OP;
    __device__ func_t pCRY_OP = CRY_OP;
    __device__ func_t pCRZ_OP = CRZ_OP;
    __device__ func_t pCU1_OP = CU1_OP;
    __device__ func_t pCU3_OP = CU3_OP;
    __device__ func_t pRXX_OP = RXX_OP;
    __device__ func_t pRZZ_OP = RZZ_OP;
    __device__ func_t pRCCX_OP = RCCX_OP;
    __device__ func_t pRC3X_OP = RC3X_OP;
    __device__ func_t pC3X_OP = C3X_OP;
    __device__ func_t pC3SQRTX_OP = C3SQRTX_OP;
    __device__ func_t pC4X_OP = C4X_OP;
    __device__ func_t pR_OP = R_OP;
    __device__ func_t pSRN_OP = SRN_OP;
    __device__ func_t pW_OP = W_OP;
    __device__ func_t pRYY_OP = RYY_OP;
    //=====================================================================================

}; // namespace SVSim
#endif
