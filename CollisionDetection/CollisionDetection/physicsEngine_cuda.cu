/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

	void copyArrayFromDevice(void *host, const void *device, int size)
	{
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
		//cudaMemcpyFromSymbol(host, device, size);
	}

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void update_dynamics(float *pos, float *velo, float *radius, float elapse, uint sphere_num)
    {
        /*thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+numParticles, d_vel4+numParticles)),
            integrate_functor(deltaTime));*/
		uint numThreads, numBlocks;
		computeGridSize(sphere_num, 256, numBlocks, numThreads);
		update_dynamics<<< numBlocks, numThreads >>>(
			(float3 *) pos, 
			(float3 *) velo, 
			radius, 
			elapse, 
			sphere_num);
		getLastCudaError("Kernel execution failed");
    }

    void calcHash(uint  *hash, uint  *index, float *pos, uint sphere_num)
    {
        uint numThreads, numBlocks;
        computeGridSize(sphere_num, 256, numBlocks, numThreads);
        calcHashD<<< numBlocks, numThreads >>>(
			hash,
			index,
			(float3 *) pos,
			sphere_num);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *pos_sorted,
                                     float *velo_sorted,
                                     uint  *hash,
                                     uint  *index,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
        reorderDataAndFindCellStartD<<< numBlocks, numThreads >>>(
            cellStart,
            cellEnd,
            (float3 *) pos_sorted,
            (float3 *)velo_sorted,
            hash,
            index,
            (float3 *) oldPos,
            (float3 *) oldVel,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

    }

    void collide(float *newVel,
                 float *pos_sorted,
                 float *velo_sorted,
		         float *radius,
		float *mass,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells)
    {

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((float3 *)newVel,
                                              (float3 *)pos_sorted,
                                              (float3 *)velo_sorted,
			radius, mass,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
                                              numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

    }


    void radixSortByHash(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

}   // extern "C"
