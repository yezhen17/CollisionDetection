/*
 * The implementation of GPU basic functions and simulation functions
 */

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "global.h"
#include "dSimulationKernel.cuh"


extern "C" {
    void cudaInit(int argc, char **argv) {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0) {
            printf("No CUDA Capable devices found, please switch to CPU mode, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, uint size) {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

	void zeroizeArray(void *devPtr, uint size) {
		checkCudaErrors(cudaMemset(devPtr, 0x0, size));
	}

    void freeArray(void *devPtr) {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync() {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyHost2Device(void *device, const void *host, int offset, int size) {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

	void copyDevice2Host(void *host, const void *device, int size) {
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	}

    void dSetupSimulation(
		SimulationEnv *h_env, 
		SimulationSphereProto *h_protos) {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(d_env, h_env, sizeof(SimulationEnv)));
		checkCudaErrors(cudaMemcpyToSymbol(d_protos, h_protos, sizeof(SimulationSphereProto)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads) {
        numThreads = min(blockSize, n);
		numBlocks = (n + numThreads - 1) / numThreads; //    iDivUp(n, numThreads);
    }

    void dHashifyAndSort(
		uint  *hashes, 
		uint  *indices, 
		float *pos, 
		uint sphere_num) {
        uint numThreads, numBlocks;
        computeGridSize(sphere_num, 256, numBlocks, numThreads);
		
		// parallelly calculate the hash value of every sphere
		hashifyKernel <<< numBlocks, numThreads >>> (
			hashes,
			indices,
			(float3 *) pos);

		getLastCudaError("Kernel execution failed: hashify");

		// use thrust radix sort to sort the hashes
		thrust::sort_by_key(
			thrust::device_ptr<uint>(hashes),
			thrust::device_ptr<uint>(hashes + sphere_num),
			thrust::device_ptr<uint>(indices));
    }

    void dCollectCells(
		uint *cell_start,
		uint *cell_end,
		uint *hash,
		uint sphere_num,
		uint max_hash_value) {
        uint numThreads, numBlocks;
        computeGridSize(sphere_num, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cell_start, 0xffffffff, max_hash_value *sizeof(uint)));

		// parallelly find out all the locations where a cell starts or ends
        collectCellsKernel <<< numBlocks, numThreads >>>(
            cell_start,
            cell_end,
            hash);
		
        getLastCudaError("Kernel execution failed: collectCells");
    }

    void dNarrowPhaseCollisionDetection(
		float *velo_delta_s,
		float *pos_s,
		float *velo_s,
		uint *types,
		uint  *indices_sorted,
		uint  *cell_start,
		uint  *cell_end,
		uint   sphere_num) {
        uint numThreads, numBlocks;
        computeGridSize(sphere_num, 64, numBlocks, numThreads);

        // execute the kernel
		collisionKernel <<< numBlocks, numThreads >>> (
			(float3 *) velo_delta_s,
			(float3 *) pos_s,
			(float3 *) velo_s,
			types,
			indices_sorted,
			cell_start,
			cell_end);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

	void dUpdateDynamics(
		float *pos_s,
		float *velo_s,
		float *velo_delta_s,
		uint *types,
		float elapse,
		uint sphere_num) {
		uint numThreads, numBlocks;
		computeGridSize(sphere_num, 256, numBlocks, numThreads);

		// parallelly update the position and velocity of each sphere
		updateDynamicsKernel << < numBlocks, numThreads >> > (
			(float3 *)pos_s,
			(float3 *)velo_s,
			(float3 *)velo_delta_s,
			types,
			elapse);

		getLastCudaError("Kernel execution failed");
	}

}   // extern "C"
