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

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
#include "device_launch_parameters.h"

 //Calculate the least significant 32 bits of the product of the least significant 24 bits of two unsigned integers.
// __device__ unsigned int __umul24(unsigned int  x, unsigned int  y) {};
//Calculate the least significant 32 bits of the product of the least significant 24 bits of two integers.
// __device__ int __mul24(int  x, int  y) {};

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;

#define GET_INDEX __mul24(blockIdx.x,blockDim.x) + threadIdx.x


struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

__global__ void update_dynamics(float3 *pos4_, float3 *velo4_, float *radius_, float elapse, uint sphere_num)
{
	uint index = GET_INDEX;
	if (index >= sphere_num) return;

	float3 pos = pos4_[index];
	float3 velo = velo4_[index];

	/*float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
	float3 velo = make_float3(velo4.x, velo4.y, velo4.z);*/
	float radius = radius_[index];

	velo += params.gravity * elapse;
	velo *= params.globalDamping;

	// new position = old position + velocity * deltaTime
	pos += velo * elapse;

	// set this to zero to disable collisions with cube sides
#if 1

	if (pos.x > 1.0f - radius)
	{
		pos.x = 1.0f - radius;
		velo.x *= params.boundaryDamping;
	}

	if (pos.x < -1.0f + radius)
	{
		pos.x = -1.0f + radius;
		velo.x *= params.boundaryDamping;
	}

	if (pos.y > 1.0f - radius)
	{
		pos.y = 1.0f - radius;
		velo.y *= params.boundaryDamping;
	}

	if (pos.z > 1.0f - radius)
	{
		pos.z = 1.0f - radius;
		velo.z *= params.boundaryDamping;
	}

	if (pos.z < -1.0f + radius)
	{
		pos.z = -1.0f + radius;
		velo.z *= params.boundaryDamping;
	}

#endif

	if (pos.y < -1.0f + radius)
	{
		pos.y = -1.0f + radius;
		velo.y *= params.boundaryDamping;
	}
    
	pos4_[index] = pos;
	velo4_[index] = velo;
}

// calculate position in uniform grid
__device__ int3 convertWorldPosToGrid(float3 world_pos)
{
	int3 grid_pos;
	grid_pos.x = floor((world_pos.x - params.worldOrigin.x) / params.cellSize.x);
	grid_pos.y = floor((world_pos.y - params.worldOrigin.y) / params.cellSize.y);
	grid_pos.z = floor((world_pos.z - params.worldOrigin.z) / params.cellSize.z);
    return grid_pos;
}



// calculate address in grid from position (clamping to edges)
__device__ uint hash_func(int3 grid_pos)
{
	grid_pos.x = grid_pos.x & (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	grid_pos.y = grid_pos.y & (params.gridSize.y - 1);
	grid_pos.z = grid_pos.z & (params.gridSize.z - 1);
	return grid_pos.x + (grid_pos.y << params.grid_exp.x) + (grid_pos.z << (params.grid_exp.x + params.grid_exp.y));
   
    // return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint *gridParticleHash, uint *gridParticleIndex, float3 *pos, uint sphere_num)
{
    uint index = GET_INDEX;

    if (index >= sphere_num) return;

    /*volatile float3 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));*/


	/*int3 gridPos = calcGridPos(pos[index]);
    uint hash = calcGridHash(gridPos);*/

	float3 world_pos = pos[index];
	int3 grid_pos = convertWorldPosToGrid(world_pos);

	uint hash = hash_func(grid_pos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
	uint   *cellEnd,          // output: cell end index
	float3 *sortedPos,        // output: sorted positions
	float3 *sortedVel,        // output: sorted velocities
	uint   *gridParticleHash, // input: sorted grid hashes
	uint   *gridParticleIndex,// input: sorted particle indices
	float3 *oldPos,           // input: sorted position array
	float3 *oldVel,           // input: sorted velocity array
	uint    sphere_num)
{
	// Handle to thread block group
	// cg::thread_block cta = cg::this_thread_block();
	//extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = GET_INDEX;

	if (index >= sphere_num) return;

	uint hash = gridParticleHash[index];
	if (index == 0 || hash != gridParticleHash[index - 1])
	{
		cellStart[hash] = index;

		if (index > 0)
			cellEnd[gridParticleHash[index - 1]] = index;
	}

	if (index == sphere_num - 1)
	{
		cellEnd[hash] = index + 1;
	}

	// Now use the sorted index to reorder the pos and vel data
	uint sortedIndex = gridParticleIndex[index];
	float3 pos = oldPos[sortedIndex];       // macro does either global read or texture fetch
	float3 vel = oldVel[sortedIndex];       // see particles_kernel.cuh

	sortedPos[index] = pos;
	sortedVel[index] = vel;

    //// handle case when no. of particles not multiple of block size
    //if (index < sphere_num)
    //{
    //    hash = gridParticleHash[index];

    //    // Load hash data into shared memory so that we can look
    //    // at neighboring particle's hash value without loading
    //    // two hash values per thread
    //    sharedHash[threadIdx.x+1] = hash;

    //    if (index > 0 && threadIdx.x == 0)
    //    {
    //        // first thread in block must load neighbor particle hash
    //        sharedHash[0] = gridParticleHash[index-1];
    //    }
    //}

    //cg::sync(cta);

    //if (index < sphere_num)
    //{
    //    // If this particle has a different cell index to the previous
    //    // particle then it must be the first particle in the cell,
    //    // so store the index of this particle in the cell.
    //    // As it isn't the first particle, it must also be the cell end of
    //    // the previous particle's cell

    //    if (index == 0 || hash != sharedHash[threadIdx.x])
    //    {
    //        cellStart[hash] = index;

    //        if (index > 0)
    //            cellEnd[sharedHash[threadIdx.x]] = index;
    //    }

    //    if (index == sphere_num - 1)
    //    {
    //        cellEnd[hash] = index + 1;
    //    }

  //      // Now use the sorted index to reorder the pos and vel data
  //      uint sortedIndex = gridParticleIndex[index];
		//float3 pos = oldPos[sortedIndex];       // macro does either global read or texture fetch
		//float3 vel = oldVel[sortedIndex];       // see particles_kernel.cuh

  //      sortedPos[index] = pos;
  //      sortedVel[index] = vel;
  //  }


}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}


// collide two spheres using DEM method
__device__
float3 collideSpheresMine(float3 posA, float3 posB,
	float3 velA, float3 velB, 
	float radiusA, float radiusB,
	float mass_c, float mass_n)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;

		float3 normVel = (dot(relVel, norm) * norm);

		// relative tangential velocity
		float3 tanVel = relVel - normVel;

		// spring force
		float deform = collideDist - dist;
		if (deform > radiusA * 2)
		{
			deform = radiusA * 2;
		}
		force = -params.spring*deform * norm;
		// dashpot (damping) force
		//force += params.damping*relVel;
		// tangential shear force
	    //force += params.shear*tanVel;
		// attraction
		//force += attraction * relPos;

		force += params.e * normVel;

	    //float3 impulse = relVel * (1.0f + params.e) * 0.5f;
		//force = dot(impulse, norm) * norm;
	}

	return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    grid_pos,
	uint    index,
	float3  pos,
	float3  vel,
	float radius_c,
	float mass_c,
	float *radius_all,
	float*mass_all,
	float3 *oldPos,
	float3 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd,
	uint   *gridParticleIndex)
{
    uint gridHash = hash_func(grid_pos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = oldPos[j];
                float3 vel2 = oldVel[j];
				uint originalIndex = gridParticleIndex[j];

				float radius_n = radius_all[originalIndex];
				float mass_n = mass_all[originalIndex];

                // collide two spheres
                force += collideSpheresMine(pos, pos2, vel, vel2, radius_c, radius_n, mass_c, mass_n);
            }
        }
    }

    return force;
}


__global__
void collideD(float3 *newVel,               // output: new velocity
	float3 *oldPos,               // input: sorted positions
	float3 *oldVel,               // input: sorted velocities
	float *radius_all,
	float *mass_all,
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    sphere_num)
{
    uint index = GET_INDEX;

    if (index >= sphere_num) return;

	uint originalIndex = gridParticleIndex[index];

    // read particle data from sorted arrays
    float3 pos = oldPos[index];
    float3 vel = oldVel[index];
	float radius_c = radius_all[originalIndex];
	float mass_c = mass_all[originalIndex];
    // get address in grid
    int3 grid_pos = convertWorldPosToGrid(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = grid_pos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, radius_c, mass_c, radius_all, mass_all, oldPos, oldVel, cellStart, cellEnd, gridParticleIndex);
            }
        }
    }

    // collide with cursor sphere
    // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

    // write new velocity back to original unsorted location
   

    newVel[originalIndex] = vel + (force / mass_c);
}

#endif
