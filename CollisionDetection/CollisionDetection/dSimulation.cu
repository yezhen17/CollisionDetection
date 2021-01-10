/*
 * The implementation of GPU basic functions and simulation functions
 * reference: CUDA 10.1 samples (particles)
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "dSimulationKernel.cuh"

typedef unsigned int uint;

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

    void allocateArray(void **dev_ptr, uint size) {
        checkCudaErrors(cudaMalloc(dev_ptr, size));
    }

	void zeroizeArray(void *dev_ptr, uint size) {
		checkCudaErrors(cudaMemset(dev_ptr, 0x0, size));
	}

    void freeArray(void *dev_ptr) {
        checkCudaErrors(cudaFree(dev_ptr));
    }

    void copyHost2Device(void *device, const void *host, int offset, int size) {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

	void copyDevice2Host(void *host, const void *device, int size) {
		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	}

    void dSetupSimulation(SimulationEnv *h_env,  SimulationSphereProto *h_protos) {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(d_env, h_env, sizeof(SimulationEnv)));
		checkCudaErrors(cudaMemcpyToSymbol(d_protos, h_protos, sizeof(SimulationSphereProto)));
    }

	void dSimulateFast(float * pos_s, float * velo_s, float * velo_delta_s, uint * types, 
		uint * hashes, uint * indices, uint * cell_start, uint * cell_end, 
		float elapse, uint sphere_num, uint max_hash_value) {
		uint num_threads, num_blocks;
		num_threads = min(256, sphere_num);
		num_blocks = (sphere_num + num_threads - 1) / num_threads;

		// first calculate the hash value of every sphere
		hashifyKernel <<< num_blocks, num_threads >>> (
			hashes,
			indices,
			(float3 *)pos_s);

		getLastCudaError("HashifyKernel execution failed.");

		// use thrust radix sort to sort the hashes
		// and we get the index mapping
		thrust::sort_by_key(
			thrust::device_ptr<uint>(hashes),
			thrust::device_ptr<uint>(hashes + sphere_num),
			thrust::device_ptr<uint>(indices));

		// set all cells to empty
		checkCudaErrors(cudaMemset(cell_start, 0xffffffff, max_hash_value * sizeof(uint)));

		// find out all the locations where a cell starts or ends
		collectCellsKernel <<< num_blocks, num_threads >>> (
			cell_start,
			cell_end,
			hashes);

		getLastCudaError("CollectCellsKernel execution failed.");

		// process collision by parallelly traversing every cell
		collisionKernel <<< num_blocks, num_threads >>> (
			(float3 *)velo_delta_s,
			(float3 *)pos_s,
			(float3 *)velo_s,
			types,
			indices,
			cell_start,
			cell_end);

		getLastCudaError("CollisionKernel execution failed.");

		// update the position and velocity of each sphere
		updateDynamicsKernel <<<  num_blocks, num_threads >>> (
			(float3 *)pos_s,
			(float3 *)velo_s,
			(float3 *)velo_delta_s,
			types,
			elapse);

		getLastCudaError("UpdateDynamicsKernel execution failed.");
	}
}
