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

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

	void copyArrayFromDevice(void *host, const void *device, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);

    void dSetupSimulation(SimulationEnv *hostParams);

    void dUpdateDynamics(float *pos, float *velo, float *velo_delta, float *radius, float elapse, uint sphere_num);

    void dHashifyAndSort(uint *hashes, uint *indices, float *pos, uint sphere_num);

    void dCollectCells(uint *cellStart,uint *cellEnd, uint *hashes, uint sphere_num, uint cell_num);

    /*void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   cell_num);*/

	void collide(float *newVel,
		float *sortedPos,
		float *sortedVel,
		float *radius,
		float *mass,
		uint  *gridParticleIndex,
		uint  *cellStart,
		uint  *cellEnd,
		uint   numParticles,
		uint   cell_num);
}
