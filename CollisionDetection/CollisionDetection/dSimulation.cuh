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
    void dUpdateDynamics(float *pos_s, float *velo_s, float *velo_delta_s, uint *types, float elapse, uint sphere_num);
    void dHashifyAndSort(uint *hashes, uint *indices, float *pos, uint sphere_num);
    void dCollectCells(uint *cell_start, uint *cell_end, uint *hashes, uint sphere_num, uint max_hash_value);
	void dNarrowPhaseCollisionDetection(float *velo_delta_s, float *pos_s, float *velo_s, uint *types, uint *indices_sorted, uint *cell_start, uint *cell_end, uint sphere_num);
}

#endif
