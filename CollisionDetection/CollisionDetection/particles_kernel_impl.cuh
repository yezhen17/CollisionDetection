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
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#define GET_INDEX __mul24(blockIdx.x,blockDim.x) + threadIdx.x

/******* Constant GPU Memory *******/
__constant__ SimulationEnv d_env; // simulation parameters in constant memory

__global__ void updateDynamicsKernel(
	float3 *pos_s, 
	float3 *velo_s, 
	float3 *velo_delta_s, 
	float *radii, 
	float elapse)
{
	uint index = GET_INDEX;
	if (index >= d_env.sphere_num) return;

	float3 pos = pos_s[index];
	float3 velo = velo_s[index];
	float3 velo_delta = velo_delta_s[index];
	float radius = radii[index];

	velo += velo_delta;
	velo += d_env.gravity * elapse;
	velo *= d_env.drag;

	// new position = old position + velocity * deltaTime
	pos += velo * elapse;

	// set this to zero to disable collisions with cube sides
#if 1

	if (pos.x > 1.0f - radius)
	{
		pos.x = 1.0f - radius;
		velo.x *= d_env.boundary_damping;
	}

	if (pos.x < -1.0f + radius)
	{
		pos.x = -1.0f + radius;
		velo.x *= d_env.boundary_damping;
	}

	if (pos.y > 1.0f - radius)
	{
		pos.y = 1.0f - radius;
		velo.y *= d_env.boundary_damping;
	}

	if (pos.z > 1.0f - radius)
	{
		pos.z = 1.0f - radius;
		velo.z *= d_env.boundary_damping;
	}

	if (pos.z < -1.0f + radius)
	{
		pos.z = -1.0f + radius;
		velo.z *= d_env.boundary_damping;
	}

#endif

	if (pos.y < -1.0f + radius)
	{
		pos.y = -1.0f + radius;
		velo.y *= d_env.boundary_damping;
	}
    
	pos_s[index] = pos;
	velo_s[index] = velo;
}

// calculate position in uniform grid
__device__ int3 convertWorldPosToGrid(float3 world_pos)
{
	int3 grid_pos;
	grid_pos.x = floor((world_pos.x - d_env.worldOrigin.x) / d_env.cell_size.x);
	grid_pos.y = floor((world_pos.y - d_env.worldOrigin.y) / d_env.cell_size.y);
	grid_pos.z = floor((world_pos.z - d_env.worldOrigin.z) / d_env.cell_size.z);
    return grid_pos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint hashFunc(int3 grid_pos)
{
	grid_pos.x = grid_pos.x & (d_env.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	grid_pos.y = grid_pos.y & (d_env.grid_size.y - 1);
	grid_pos.z = grid_pos.z & (d_env.grid_size.z - 1);
	return grid_pos.x + (grid_pos.y << d_env.grid_exp.x) + (grid_pos.z << (d_env.grid_exp.x + d_env.grid_exp.y));
   
    // return __umul24(__umul24(gridPos.z, d_env.grid_size.y), d_env.grid_size.x) + __umul24(gridPos.y, d_env.grid_size.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void hashifyKernel(
	uint *hashes, 
	uint *indices_to_sort,
	float3 *pos)
{
    uint index = GET_INDEX;

    if (index >= d_env.sphere_num) return;

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
	uint   *cell_start,        // output: cell start index
	uint   *cell_end,          // output: cell end index
	uint   *hashes)
{
	uint index = GET_INDEX;
	if (index >= d_env.sphere_num) return;

	uint hash = hashes[index];
	if (index == 0)
	{
		cell_start[hash] = index;
	}
	else if (hash != hashes[index - 1])
	{
		cell_start[hash] = index;
		cell_end[hashes[index - 1]] = index;
	}
	if (index == d_env.sphere_num - 1)
	{
		cell_end[hash] = index + 1;
	}
}

// Use the DEM method adapted to various masses and restitudes
__device__ float3 collisionAtomic(
	float3 pos_c, 
	float3 pos_n,
	float3 velo_c, 
	float3 velo_n, 
	float radius_c, 
	float radius_n,
	float mass_c, 
	float mass_n)
{
	float3 displacement = pos_n - pos_c;

	float distance = length(displacement);
	float radius_sum = radius_c + radius_n;

	float3 force = make_float3(0.0f);

	if (distance < radius_sum)
	{
		float3 normal = displacement / distance;

		// relative velocity
		float3 velo_relative = velo_n - velo_c;

		float3 velo_normal = (dot(velo_relative, normal) * normal);

		// relative tangential velocity
		float3 velo_tangent = velo_relative - velo_normal;

		// spring force
		float deform = radius_sum - distance;
		if (deform > radius_c * 2)
		{
			deform = radius_c * 2;
		}
		force = -(d_env.spring * deform) * normal;
		// dashpot (damping) force
		//force += d_env.damping*velo_relative;
		// tangential shear force
	    //force += d_env.shear*velo_tangent;

		force += d_env.e * velo_normal;

	    //float3 impulse = velo_relative * (1.0f + d_env.e) * 0.5f;
		//force = dot(impulse, normal) * normal;
	}

	return force;
}

// collide a particle against all other particles in a given cell
__device__ float3 collisionInCell(
	int3 grid_pos,
	uint index_self,
	float3 pos_c,
	float3 velo_c,
	float radius_c,
	float mass_c,
	float *radii,
	float *masses,
	float3 *pos_s,
	float3 *velo_s,
	uint *cell_start,
	uint *cell_end,
	uint *indices_sorted)
{
    uint hash = hashFunc(grid_pos);

    // get start of bucket for this cell
    uint index_cell_start = cell_start[hash];

    float3 force = make_float3(0.0f);

    if (index_cell_start != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint index_cell_end = cell_end[hash];
		for (uint j = index_cell_start; j < index_cell_end; ++j)
        {
			uint index_origin = indices_sorted[j];
            if (index_origin != index_self)                // check not colliding with self
            {	
				float3 pos_n = pos_s[index_origin];
				float3 vel_n = velo_s[index_origin];
				float radius_n = radii[index_origin];
				float mass_n = masses[index_origin];
                // collide two spheres
                force += collisionAtomic(pos_c, pos_n, velo_c, vel_n, radius_c, radius_n, mass_c, mass_n);
            }
        }
    }
    return force;
}

__global__ void collisionKernel(
	float3 *velo_delta_s,               // output: new velocity
	float3 *pos_s,               // input: sorted positions
	float3 *velo_s,               // input: sorted velocities
	float *radii,
	float *masses,
	float *rest_s,
	uint   *indices_sorted,    // input: sorted particle indices_sorted
	uint   *cell_start,
	uint   *cell_end)
{
    uint index = GET_INDEX;
    if (index >= d_env.sphere_num) return;

	// Now use the sorted index to reorder the pos and vel data
	uint index_origin = indices_sorted[index];
    float3 pos = pos_s[index_origin];
    float3 velo = velo_s[index_origin];
	float radius_c = radii[index_origin];
	float mass_c = masses[index_origin];
	float rest_c = rest_s[index_origin];
    // get address in grid
    int3 grid_pos = convertWorldPosToGrid(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

	// need not deal with out-of-boundary neighbors because of hashing
	for (int z = -1; z <= 1; ++z)
    {
		for (int y = -1; y <= 1; ++y)
        {
			for (int x = -1; x <= 1; ++x)
            {
                int3 adjacent_pos = grid_pos + make_int3(x, y, z);
                force += collisionInCell(
					adjacent_pos,
					index_origin, 
					pos, 
					velo, 
					radius_c, 
					mass_c, 
					radii, 
					masses, 
					pos_s, 
					velo_s, 
					cell_start, 
					cell_end, 
					indices_sorted);
            }
        }
    }

    // write velocity change
	velo_delta_s[index_origin] = force / mass_c;
}

#endif
