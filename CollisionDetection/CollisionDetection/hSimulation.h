/*
 * This file provides the CPU (serial) implementation of the collision detection algorithm based on spatial subdivision/brutal force algorithm
 */

#ifndef HSIMULATION_H
#define HSIMULATION_H

#include <algorithm>
#include <helper_math.h>
#include <glm/glm.hpp>

#include "global.h"
#include "environment.h"
#include "mortonEncode.cuh"

SimulationEnv *h_env; // Environment parameters
SimulationSphereProto *h_protos; // Sphere parameters (fixed throughout simulation)

int3 h_neighboorhood_3[27] = {
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
int3 hConvertWorldPosToGrid(float3 world_pos) {
	int3 grid_pos;
	grid_pos.x = (int)floor(world_pos.x / h_env->cell_size);
	grid_pos.y = (int)floor(world_pos.y / h_env->cell_size);
	grid_pos.z = (int)floor(world_pos.z / h_env->cell_size);
	return grid_pos;
}

uint hHashFunc(int3 grid_pos) {
	return hMortonEncode3D(grid_pos);
	//grid_pos.x = grid_pos.x & (h_env->grid_size.x - 1);  // wrap grid, assumes size is power of 2
	//grid_pos.y = grid_pos.y & (h_env->grid_size.y - 1);
	//grid_pos.z = grid_pos.z & (h_env->grid_size.z - 1);
	//return grid_pos.x + (grid_pos.y << h_env->grid_exp.x) + (grid_pos.z << (h_env->grid_exp.x + h_env->grid_exp.y));
}

void hSetupSimulation(SimulationEnv *env, SimulationSphereProto *stats) {
	h_env = env;
	h_protos = stats;
}

void hHashifyAndSort(
	uint *hashes,
	uint *indices_to_sort,
	float3 *pos) {
	std::vector<std::pair<uint, uint>> hashes_and_indices;
	for (uint i = 0; i < h_env->sphere_num; ++i) {
		int3 grid_pos = hConvertWorldPosToGrid(pos[i]);
		uint hash = hHashFunc(grid_pos);

		// store grid hash and particle index
		hashes_and_indices.push_back(std::make_pair(hash, i));
	}

	// sort and then reassign
	std::stable_sort(hashes_and_indices.begin(), hashes_and_indices.end(),
		[](std::pair<uint, uint> const &a, std::pair<uint, uint> const &b) {
		return a.first < b.first;
	});
	int i = 0;
	for (auto &it : hashes_and_indices) {
		hashes[i] = it.first;
		indices_to_sort[i] = it.second;
		++i;
	}
}

void hCollectCells(
	uint *cell_start,
	uint *cell_end,
	uint *hashes) {
	memset(cell_start, (unsigned char)0xff, h_env->max_hash_value * sizeof(uint));
	for (uint i = 0; i < h_env->sphere_num; ++i) {
		uint hash = hashes[i];
		if (i == 0) {
			cell_start[hash] = i;
		}
		else if (hash != hashes[i - 1]) {
			cell_start[hash] = i;
			cell_end[hashes[i - 1]] = i;
		}
		if (i == h_env->sphere_num - 1) {
			cell_end[hash] = i + 1;
		}
	}
}

// Use the DEM method adapted to various masses and restitudes
float3 hCollisionAtomic(
	float3 pos_c,
	float3 pos_n,
	float3 velo_c,
	float3 velo_n,
	float radius_c,
	float mass_c,
	uint type_c,
	uint type_n) {
	float radius_n = h_protos->radii[type_n];
	float mass_n = h_protos->masses[type_n];

	float3 displacement = pos_n - pos_c;
	float distance = length(displacement);
	float radius_sum = radius_c + radius_n;

	float3 force = make_float3(0.0f);

	if (distance < radius_sum) {
		float3 normal = displacement / distance;

		// relative velocity (neighbor relative to center)
		float3 velo_relative = velo_n - velo_c;

		// relative normal velocity 
		float3 velo_normal = (dot(velo_relative, normal) * normal);

		// relative tangential velocity
		float3 velo_tangent = velo_relative - velo_normal;

		// deformation
		float deform = radius_sum - distance;
		if (deform > radius_c * 2) {
			deform = radius_c * 2;
		}

		// spring force (linear spring model)
		force -= (h_env->stiffness * deform) * normal;

		// damping force (dashpot model)
		// in some models the direction is relative normal direction; 
		// in others the direction is relative direction
		// += because the relative velocity is neighbor w.r.t. center
		float damping = h_protos->damping[type_c][type_n];
		float mass_sqrt = sqrtf(mass_c*mass_n / (mass_c + mass_n));
		force += (h_env->damping * damping * mass_sqrt) * velo_relative;

		// tangential friction force (optional, defaults to zero)
		float force_normal = -dot(force, normal);
		force += (h_env->friction * force_normal) * velo_tangent;
	}

	return force;
}

void hNarrowPhaseCollisionDetection(
	float3 *accel_s, 
	float3 *pos_s,  
	float3 *velo_s, 
	uint *types,
	uint *indices_sorted,  
	uint *cell_start,
	uint *cell_end) {
	for (uint i = 0; i < h_env->sphere_num; ++i) {
		// Now use the sorted index to reorder the pos and vel data
		uint index_origin_c = indices_sorted[i];
		float3 pos_c = pos_s[index_origin_c];
		float3 velo_c = velo_s[index_origin_c];
		uint type_c = types[index_origin_c];
		float radius_c = h_protos->radii[type_c];
		float mass_c = h_protos->masses[type_c];
		// get address in grid
		int3 grid_pos_c = hConvertWorldPosToGrid(pos_c);

		// examine neighbouring cells
		float3 force = make_float3(0.0f);

		// need not deal with out-of-boundary neighbors because of hashing
		for (uint i = 0; i < 27; ++i) {
			uint hash = hHashFunc(grid_pos_c + h_neighboorhood_3[i]);

			// get start of bucket for this cell
			uint index_cell_start = cell_start[hash];

			if (index_cell_start != 0xffffffff) {
				// iterate over particles in this cell
				uint index_cell_end = cell_end[hash];
				for (uint j = index_cell_start; j < index_cell_end; ++j) {
					uint index_origin_n = indices_sorted[j];

					// prevent colliding with itself
					if (index_origin_n != index_origin_c) {
						float3 pos_n = pos_s[index_origin_n];
						float3 vel_n = velo_s[index_origin_n];
						uint type_n = types[index_origin_n];
						force += hCollisionAtomic(pos_c, pos_n, velo_c, vel_n, radius_c, mass_c, type_c, type_n);
					}
				}
			}
		}

		// write velocity change
		accel_s[index_origin_c] = force / mass_c;
	}
}

void hUpdateDynamics(
	float3 *pos_s,
	float3 *velo_s,
	float3 *accel_s,
	uint *types,
	float elapse) {
	for (uint i = 0; i < h_env->sphere_num; ++i) {
		float3 pos = pos_s[i];
		float3 velo = velo_s[i];
		float3 accel = accel_s[i];
		uint type = types[i];
		float radius = h_protos->radii[type];

		velo += accel * elapse;
		velo += h_env->gravity * elapse;
		velo *= h_env->drag;

		// new position = old position + velocity * deltaTime
		pos += velo * elapse;

		// collide with boundaries (directly use restitution)
		float restitution = -h_protos->restitution[type][type];
		float3 max_corner = h_env->max_corner;
		float3 min_corner = h_env->min_corner;
		if (pos.x > max_corner.x - radius) {
			pos.x = max_corner.x - radius;
			velo.x *= restitution;
		}
		if (pos.x < min_corner.x + radius) {
			pos.x = min_corner.x + radius;
			velo.x *= restitution;
		}
		if (pos.y > max_corner.y - radius) {
			pos.y = max_corner.y - radius;
			velo.y *= restitution;
		}
		if (pos.y < min_corner.y + radius) {
			pos.y = min_corner.y + radius;
			velo.y *= restitution;
		}
		if (pos.z > max_corner.z - radius) {
			pos.z = max_corner.z - radius;
			velo.z *= restitution;
		}
		if (pos.z < min_corner.z + radius) {
			pos.z = min_corner.z + radius;
			velo.z *= restitution;
		}
		pos_s[i] = pos;
		velo_s[i] = velo;
	}
}

// fast collision detection algorithm based on space subdivision
void hSimulateFast(
	float *pos_s,
	float *velo_s,
	float *velo_delta_s,
	uint *types,
	uint *hashes,
	uint *indices,
	uint *cell_start,
	uint *cell_end,
	float elapse) {
	hHashifyAndSort(
		hashes,
		indices,
		(float3 *)pos_s);

	hCollectCells(
		cell_start,
		cell_end,
		hashes);

	hNarrowPhaseCollisionDetection(
		(float3 *)velo_delta_s,
		(float3 *)pos_s,
		(float3 *)velo_s,
		types,
		indices,
		cell_start,
		cell_end);

	hUpdateDynamics(
		(float3 *)pos_s,
		(float3 *)velo_s,
		(float3 *)velo_delta_s,
		types,
		elapse);
}

// brutal force collision detection
void hSimulateBrutal(
	float *pos_s,
	float *velo_s,
	float *accel_s,
	uint *types,
	float elapse) {
	float3 *pos_s_3 = (float3 *)pos_s;
	float3 *velo_s_3 = (float3 *)velo_s;
	float3 *accel_s_3 = (float3 *)accel_s;
	for (uint i = 0; i < h_env->sphere_num; ++i) {
		float3 force = make_float3(0.0f);
		float3 pos_c = pos_s_3[i];
		float3 velo_c = velo_s_3[i];
		uint type_c = types[i];
		float radius_c = h_protos->radii[type_c];
		float mass_c = h_protos->masses[type_c];
		for (uint j = 0; j < h_env->sphere_num; ++j) {
			if (j != i) {
				float3 pos_n = pos_s_3[j];
				float3 velo_n = velo_s_3[j];
				uint type_n = types[j];
				float radius_n = h_protos->radii[type_n];
				float mass_n = h_protos->masses[type_n];
				force += hCollisionAtomic(pos_c, pos_n, velo_c, velo_n, radius_c, radius_n, mass_c, mass_n);
			}
		}
		accel_s_3[i] = force / mass_c;
	}
	hUpdateDynamics(pos_s_3, velo_s_3, accel_s_3, types, elapse);
}

#endif // !HSIMULATION_H

