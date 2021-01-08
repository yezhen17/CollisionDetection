/*
 * This file defines a struct of simulation environment, including all kinds of parameters
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector_types.h>

#include "global.h"

// simulation environment
struct SimulationEnv {
	uint3 grid_exp;
	float3 gravity;
	float drag;
	float max_radius;

	uint3 grid_size;
	uint cell_num;
	float3 cell_size;

	uint sphere_num;

	float stiffness;
	float damping;
	float friction;
	float boundary_damping;

	float3 min_corner;
	float3 max_corner;

	float e;
};

#endif
