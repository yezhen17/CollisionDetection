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
    void dCollectCells(uint *cell_start, uint *cell_end, uint *hashes, uint sphere_num, uint cell_num);
	void dNarrowPhaseCollisionDetection(float *velo_delta_s, float *pos_s, float *velo_s, uint *types, uint *indices_sorted, uint *cell_start, uint *cell_end, uint sphere_num);
}
