/*
 * This file defines a struct of simulation environment, including all kinds of parameters
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector_types.h>

typedef unsigned int uint;

// simulation environment
typedef struct SimulationEnv {
	uint sphere_num;
	float cell_size;
	uint max_hash_value;

	// for boundary collision
	float3 min_corner;
	float3 max_corner;

	// physical environment
	float3 gravity; 
	float drag; // global velocity decay factor
	float stiffness; // assume stiffness factor is universal
	float damping; // damping factor (the material irrelevant component)
	float friction; // assume friction factor is universal

} SimulationEnv;

// sphere prototypes for getting radius, mass, damping and restitution 
typedef struct SimulationSphereProto {
	float radii[4];
	float masses[4];
	float damping[4][4];
	float restitution[4][4];
} SimulationSphereProto;

#endif
