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


/******* Constant GPU Memory *******/
__constant__ SimulationEnv env; // simulation parameters in constant memory

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

        vel += env.gravity * deltaTime;
        vel *= env.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - env.max_radius)
        {
            pos.x = 1.0f - env.max_radius;
            vel.x *= env.boundaryDamping;
        }

        if (pos.x < -1.0f + env.max_radius)
        {
            pos.x = -1.0f + env.max_radius;
            vel.x *= env.boundaryDamping;
        }

        if (pos.y > 1.0f - env.max_radius)
        {
            pos.y = 1.0f - env.max_radius;
            vel.y *= env.boundaryDamping;
        }

        if (pos.z > 1.0f - env.max_radius)
        {
            pos.z = 1.0f - env.max_radius;
            vel.z *= env.boundaryDamping;
        }

        if (pos.z < -1.0f + env.max_radius)
        {
            pos.z = -1.0f + env.max_radius;
            vel.z *= env.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + env.max_radius)
        {
            pos.y = -1.0f + env.max_radius;
            vel.y *= env.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

__global__ void updateDynamicsKernel(float3 *pos4_, float3 *velo4_, float3 *velo_delta, float *radius_, float elapse, uint sphere_num)
{
	uint index = GET_INDEX;
	if (index >= sphere_num) return;

	float3 pos = pos4_[index];
	float3 velo = velo4_[index];

	/*float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
	float3 velo = make_float3(velo4.x, velo4.y, velo4.z);*/
	float radius = radius_[index];

	velo += velo_delta[index];

	velo += env.gravity * elapse;
	velo *= env.globalDamping;

	// new position = old position + velocity * deltaTime
	pos += velo * elapse;

	// set this to zero to disable collisions with cube sides
#if 1

	if (pos.x > 1.0f - radius)
	{
		pos.x = 1.0f - radius;
		velo.x *= env.boundaryDamping;
	}

	if (pos.x < -1.0f + radius)
	{
		pos.x = -1.0f + radius;
		velo.x *= env.boundaryDamping;
	}

	if (pos.y > 1.0f - radius)
	{
		pos.y = 1.0f - radius;
		velo.y *= env.boundaryDamping;
	}

	if (pos.z > 1.0f - radius)
	{
		pos.z = 1.0f - radius;
		velo.z *= env.boundaryDamping;
	}

	if (pos.z < -1.0f + radius)
	{
		pos.z = -1.0f + radius;
		velo.z *= env.boundaryDamping;
	}

#endif

	if (pos.y < -1.0f + radius)
	{
		pos.y = -1.0f + radius;
		velo.y *= env.boundaryDamping;
	}
    
	pos4_[index] = pos;
	velo4_[index] = velo;
}

// calculate position in uniform grid
__device__ int3 convertWorldPosToGrid(float3 world_pos)
{
	int3 grid_pos;
	grid_pos.x = floor((world_pos.x - env.worldOrigin.x) / env.cellSize.x);
	grid_pos.y = floor((world_pos.y - env.worldOrigin.y) / env.cellSize.y);
	grid_pos.z = floor((world_pos.z - env.worldOrigin.z) / env.cellSize.z);
    return grid_pos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint hashFunc(int3 grid_pos)
{
	grid_pos.x = grid_pos.x & (env.gridSize.x - 1);  // wrap grid, assumes size is power of 2
	grid_pos.y = grid_pos.y & (env.gridSize.y - 1);
	grid_pos.z = grid_pos.z & (env.gridSize.z - 1);
	return grid_pos.x + (grid_pos.y << env.grid_exp.x) + (grid_pos.z << (env.grid_exp.x + env.grid_exp.y));
   
    // return __umul24(__umul24(gridPos.z, env.gridSize.y), env.gridSize.x) + __umul24(gridPos.y, env.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void hashifyKernel(uint *hashes, uint *indices_to_sort, float3 *pos, uint sphere_num)
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

	uint hash = hashFunc(grid_pos);

    // store grid hash and particle index
	hashes[index] = hash;
	indices_to_sort[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void collectCellsKernel(
	uint   *cellStart,        // output: cell start index
	uint   *cellEnd,          // output: cell end index
	uint   *hashes)
{
	uint index = GET_INDEX;
	if (index >= env.sphere_num) return;

	uint hash = hashes[index];
	if (index == 0 || hash != hashes[index - 1])
	{
		cellStart[hash] = index;

		if (index > 0)
			cellEnd[hashes[index - 1]] = index;
	}
	if (index == env.sphere_num - 1)
	{
		cellEnd[hash] = index + 1;
	}
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
        force = -env.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += env.damping*relVel;
        // tangential shear force
        force += env.shear*tanVel;
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
		force = -env.spring*deform * norm;
		// dashpot (damping) force
		//force += env.damping*relVel;
		// tangential shear force
	    //force += env.shear*tanVel;
		// attraction
		//force += attraction * relPos;

		force += env.e * normVel;

	    //float3 impulse = relVel * (1.0f + env.e) * 0.5f;
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
	uint   *indices_sorted)
{
    uint gridHash = hashFunc(grid_pos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];
        for (uint j=startIndex; j<endIndex; j++)
        {
			uint index_sorted = indices_sorted[j];
            if (index_sorted != index)                // check not colliding with self
            {	
				float3 pos2 = oldPos[index_sorted];
				float3 vel2 = oldVel[index_sorted];
				float radius_n = radius_all[index_sorted];
				float mass_n = mass_all[index_sorted];

                // collide two spheres
                force += collideSpheresMine(pos, pos2, vel, vel2, radius_c, radius_n, mass_c, mass_n);
            }
        }
    }

    return force;
}


__global__
void collideD(float3 *velo_delta,               // output: new velocity
	float3 *oldPos,               // input: sorted positions
	float3 *oldVel,               // input: sorted velocities
	float *radius_all,
	float *mass_all,
              uint   *indices_sorted,    // input: sorted particle indices_sorted
              uint   *cellStart,
              uint   *cellEnd,
              uint    sphere_num)
{
    uint index = GET_INDEX;
    if (index >= sphere_num) return;

	// Now use the sorted index to reorder the pos and vel data
	uint index_sorted = indices_sorted[index];
	//float3 pos = oldPos[index_sorted];       // macro does either global read or texture fetch
	//float3 vel = oldVel[index_sorted];       // see particles_kernel.cuh

	//sortedPos[index] = pos;
	//sortedVel[index] = vel;


    // read particle data from sorted arrays
    float3 pos = oldPos[index_sorted];
    float3 vel = oldVel[index_sorted];
	float radius_c = radius_all[index_sorted];
	float mass_c = mass_all[index_sorted];
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
                force += collideCell(neighbourPos, index_sorted, pos, vel, radius_c, mass_c, radius_all, mass_all, oldPos, oldVel, cellStart, cellEnd, indices_sorted);
            }
        }
    }

    // collide with cursor sphere
    // force += collideSpheres(pos, env.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), env.max_radius, env.colliderRadius, 0.0f);

    // write new velocity back to original unsorted location
   

	velo_delta[index_sorted] = force / mass_c;
}

#endif
