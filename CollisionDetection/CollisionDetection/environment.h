/*
 * This file defines a struct of simulation environment, including all kinds of parameters
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector_types.h>

typedef unsigned int uint;

// simulation environment
struct SimulationEnv {
	float3 gravity;
	float drag;
	float max_radius;

	uint max_hash_value;
	float cell_size;

	uint sphere_num;

	float stiffness;
	float damping;
	float friction;
	float boundary_damping;

	float3 min_corner;
	float3 max_corner;
};

#endif
