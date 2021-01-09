/*
 * This file provides the kernel code for the collision detection algorithm based on spatial subdivision
 */

#ifndef DSIMULATIONKERNEL_CUH
#define DSIMULATIONKERNEL_CUH

#include <stdio.h>
#include <math.h>
#include <helper_math.h>
#include <math_constants.h>
#include <device_launch_parameters.h>

#include "environment.h"
#include "sphere.h"
#include "mortonEncode.cuh"

#define GET_INDEX __mul24(blockIdx.x,blockDim.x) + threadIdx.x

 /******* Constant GPU Memory *******/
__constant__ SimulationEnv d_env; // Environment parameters
__constant__ SimulationSphereProto d_protos; // Sphere parameters (fixed throughout simulation)
__constant__ int3 neighboorhood_3[27] = {
	-1, -1, -1,
	 0, -1, -1,
	 1, -1, -1,
	-1,  0, -1,
	 0,  0, -1,
	 1,  0, -1,
	-1,  1, -1,
	 0,  1, -1,
	 1,  1, -1,
	-1, -1,  0,
	 0, -1,  0,
	 1, -1,  0,
	-1,  0,  0,
	 0,  0,  0,
	 1,  0,  0,
	-1,  1,  0,
	 0,  1,  0,
	 1,  1,  0,
	-1, -1,  1,
	 0, -1,  1,
	 1, -1,  1,
	-1,  0,  1,
	 0,  0,  1,
	 1,  0,  1,
	-1,  1,  1,
	 0,  1,  1,
	 1,  1,  1,
};

// calculate position in uniform grid
__device__ int3 convertWorldPosToGrid(float3 world_pos) {
	int3 grid_pos;
	grid_pos.x = floor(world_pos.x / d_env.cell_size.x);
	grid_pos.y = floor(world_pos.y / d_env.cell_size.y);
	grid_pos.z = floor(world_pos.z / d_env.cell_size.z);
	return grid_pos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint hashFunc(int3 grid_pos) {
	//grid_pos.x = grid_pos.x & (d_env.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	//grid_pos.y = grid_pos.y & (d_env.grid_size.y - 1);
	//grid_pos.z = grid_pos.z & (d_env.grid_size.z - 1);
	//return grid_pos.x + (grid_pos.y << d_env.grid_exp.x) + (grid_pos.z << (d_env.grid_exp.x + d_env.grid_exp.y));

	// use morton encoding for more coherent memory access
	return dMortonEncode3D(grid_pos);

	// return __umul24(__umul24(gridPos.z, d_env.grid_size.y), d_env.grid_size.x) + __umul24(gridPos.y, d_env.grid_size.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__ void hashifyKernel(
	uint *hashes,
	uint *indices_to_sort,
	float3 *pos) {
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
	uint   *hashes) {
	uint index = GET_INDEX;
	if (index >= d_env.sphere_num) return;

	uint hash = hashes[index];
	if (index == 0) {
		cell_start[hash] = index;
	}
	else if (hash != hashes[index - 1]) {
		cell_start[hash] = index;
		cell_end[hashes[index - 1]] = index;
	}
	if (index == d_env.sphere_num - 1) {
		cell_end[hash] = index + 1;
	}
}

// TODO
// Use the DEM method adapted to various masses and restitudes
__device__ float3 collisionAtomic(
	float3 pos_c,
	float3 pos_n,
	float3 velo_c,
	float3 velo_n,
	float radius_c,
	float radius_n,
	float mass_c,
	float mass_n) {
	float3 displacement = pos_n - pos_c;

	float distance = length(displacement);
	float radius_sum = radius_c + radius_n;

	float3 force = make_float3(0.0f);

	if (distance < radius_sum) {
		float3 normal = displacement / distance;

		// relative velocity
		float3 velo_relative = velo_n - velo_c;

		float3 velo_normal = (dot(velo_relative, normal) * normal);

		// relative tangential velocity
		float3 velo_tangent = velo_relative - velo_normal;

		// stiffness force
		float deform = radius_sum - distance;
		if (deform > radius_c * 2) {
			deform = radius_c * 2;
		}
		force = -(d_env.stiffness * deform) * normal;

		// tangential friction force
		force += d_env.friction*velo_tangent;

		force += d_env.damping * velo_normal;

		//float3 impulse = velo_relative * (1.0f + d_env.e) * 0.5f;
		//force = dot(impulse, normal) * normal;
	}

	return force;
}

__global__ void collisionKernel(
	float3 *velo_delta_s,               // output: new velocity
	float3 *pos_s,               // input: sorted positions
	float3 *velo_s,               // input: sorted velocities
	uint *types,
	uint   *indices_sorted,    // input: sorted particle indices_sorted
	uint   *cell_start,
	uint   *cell_end) {
	uint index = GET_INDEX;
	if (index >= d_env.sphere_num) return;

	// Now use the sorted index to reorder the pos and vel data
	uint index_origin_c = indices_sorted[index];
	float3 pos_c = pos_s[index_origin_c];
	float3 velo_c = velo_s[index_origin_c];
	uint type_c = types[index_origin_c];
	float radius_c = d_protos.radii[type_c];
	float mass_c = d_protos.masses[type_c];

	// get address in grid
	int3 grid_pos_c = convertWorldPosToGrid(pos_c);

	// examine neighbouring cells
	float3 force = make_float3(0.0f);

	// need not deal with out-of-boundary neighbors because of hashing
	for (uint i = 0; i < 27; ++i) {
		uint hash = hashFunc(grid_pos_c + neighboorhood_3[i]);

		// get start of bucket for this cell
		uint index_cell_start = cell_start[hash];

		if (index_cell_start != 0xffffffff) {
			// iterate over particles in this cell
			uint index_cell_end = cell_end[hash];
			for (uint j = index_cell_start; j < index_cell_end; ++j) {
				uint index_origin_n = indices_sorted[j];
				// prevent collision with itself
				if (index_origin_n != index_origin_c) {
					float3 pos_n = pos_s[index_origin_n];
					float3 vel_n = velo_s[index_origin_n];
					uint type_n = types[index_origin_n];
					float radius_n = d_protos.radii[type_n];
					float mass_n = d_protos.masses[type_n];
					// collide two spheres
					force += collisionAtomic(pos_c, pos_n, velo_c, vel_n, radius_c, radius_n, mass_c, mass_n);
				}
			}
		}
	}

	// write velocity change
	velo_delta_s[index_origin_c] = force / mass_c;
}

__global__ void updateDynamicsKernel(
	float3 *pos_s,
	float3 *velo_s,
	float3 *velo_delta_s,
	uint *types,
	float elapse) {
	uint index = GET_INDEX;
	if (index >= d_env.sphere_num) return;

	float3 pos = pos_s[index];
	float3 velo = velo_s[index];
	float3 velo_delta = velo_delta_s[index];
	uint type = types[index];
	float radius = d_protos.radii[type];

	velo += velo_delta;
	velo += d_env.gravity * elapse;
	velo *= d_env.drag;

	// new position = old position + velocity * deltaTime
	pos += velo * elapse;

	float3 max_corner = d_env.max_corner;
	float3 min_corner = d_env.min_corner;

	// TODO
	if (pos.x > max_corner.x - radius) {
		pos.x = max_corner.x - radius;
		velo.x *= d_env.boundary_damping;
	}

	if (pos.x < min_corner.x + radius) {
		pos.x = min_corner.x + radius;
		velo.x *= d_env.boundary_damping;
	}

	if (pos.y > max_corner.y - radius) {
		pos.y = max_corner.y - radius;
		velo.y *= d_env.boundary_damping;
	}

	if (pos.y < min_corner.y + radius) {
		pos.y = min_corner.y + radius;
		velo.y *= d_env.boundary_damping;
	}

	if (pos.z > max_corner.z - radius) {
		pos.z = max_corner.z - radius;
		velo.z *= d_env.boundary_damping;
	}

	if (pos.z < min_corner.z + radius) {
		pos.z = min_corner.z + radius;
		velo.z *= d_env.boundary_damping;
	}

	pos_s[index] = pos;
	velo_s[index] = velo;
}

#endif
