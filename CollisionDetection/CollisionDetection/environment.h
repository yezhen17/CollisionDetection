/*
 * This file defines a struct of simulation environment, including all kinds of parameters
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector_types.h>

typedef unsigned int uint;


// simulation environment
struct SimulationEnv {
	uint sphere_num;
	float cell_size;
	uint max_hash_value;

	// for boundary collision
	float3 min_corner;
	float3 max_corner;

	// physical environment
	float3 gravity; 
	float drag; // global velocity decay factor
	float stiffness;
	float damping;
	float friction;
	float boundary_damping;
};

#endif
