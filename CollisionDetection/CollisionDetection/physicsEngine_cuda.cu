/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

extern "C" {
    void cudaInit(int argc, char **argv) {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0) {
            printf("No CUDA Capable devices found, exiting...\n");
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

    void copyArrayToDevice(void *device, const void *host, int offset, int size) {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

	void copyArrayFromDevice(void *host, const void *device, int size) {
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
		//cudaMemcpyFromSymbol(host, device, size);
	}

    void dSetupSimulation(
		SimulationEnv *h_env, 
		SimulationSphereStats *h_stats) {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(d_env, h_env, sizeof(SimulationEnv)));
		checkCudaErrors(cudaMemcpyToSymbol(d_stats, h_stats, sizeof(SimulationSphereStats)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads) {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
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
		updateDynamicsKernel <<< numBlocks, numThreads >>> (
			(float3 *)pos_s,
			(float3 *)velo_s,
			(float3 *)velo_delta_s,
			types,
			elapse);

		getLastCudaError("Kernel execution failed");
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
		uint cell_num) {
        uint numThreads, numBlocks;
        computeGridSize(sphere_num, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cell_start, 0xffffffff, cell_num *sizeof(uint)));

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

}   // extern "C"
