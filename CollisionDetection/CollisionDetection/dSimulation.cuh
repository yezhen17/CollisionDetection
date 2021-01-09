/*
 * The header file of GPU basic functions and simulation functions
 */

#ifndef DSIMULATION_CUH
#define DSIMULATION_CUH

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, uint size);
	void zeroizeArray(void *devPtr, uint size);
    void freeArray(void *devPtr);

    void threadSync();

	void copyDevice2Host(void *host, const void *device, int size);
    void copyHost2Device(void *device, const void *host, int offset, int size);

    void dSetupSimulation(SimulationEnv *h_env, SimulationSphereProto *h_protos);
	void dSimulateFast(float *pos_s, float *velo_s, float *velo_delta_s, uint *types, 
		uint *hashes, uint *indices, uint *cell_start, uint *cell_end, float elapse, uint sphere_num, uint max_hash_value);
}

#endif
