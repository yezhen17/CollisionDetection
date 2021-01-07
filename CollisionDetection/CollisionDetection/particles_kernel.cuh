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

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"

typedef unsigned int uint;


// simulation parameters
struct SimulationEnv
{
	uint3 grid_exp;

    float3 gravity;
    float drag;
    float max_radius;

    uint3 grid_size;
    uint cell_num;
    float3 worldOrigin;
    float3 cell_size;

    uint sphere_num;

    float stiffness;
    float damping;
    float friction;
    float boundary_damping;

	float e;
};

struct SimulationSphereStats
{
	float radii[7];
	float masses[7];
};

#endif
