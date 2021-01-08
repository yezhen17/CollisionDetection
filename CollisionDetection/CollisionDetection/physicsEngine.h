/*
 * This file defines a physics engine that supports simulation based on physics rules
 */

#ifndef PHYSICSENGINE_H
#define PHYSICSENGINE_H

#include <glm/glm.hpp>
#include <helper_functions.h>
#include <vector_functions.h>

# include "global.h"
#include "sphere.h"
#include "environment.h"

class PhysicsEngine {
public:
	PhysicsEngine(uint sphere_num, 
		glm::vec3 origin, 
		glm::vec3 room_size, 
		uint grid_size=GRID_SIZE, 
		bool gpu_mode= GPU_MODE);

	~PhysicsEngine();

	// output  updated sphere positions
	float* outputPos();

	// simulate the collision and motion of all spheres in given amount of time (elapse) 
	void update(float elapse);

	uint *getSphereType() { return h_type_; }

protected:
	// initialize the environment
	void initEnvironment();

	// initialize the type, position and velocity of all spheres
	void initSpheres();

protected:
	// use GPU for calculation or not
	bool gpu_mode_;

	uint sphere_num_;
	// CPU data
	float *h_pos_;              // particle positions
	float *h_velo_;              // particle velocities
	float *h_velo_delta_;

	float *h_color_;
	uint *h_type_;

	uint  *h_hash_; // grid hash value for each particle
	uint  *h_index_sorted_;// particle index for each particle
	uint  *h_cell_start_;        // index of start of each cell in sorted list
	uint  *h_cell_end_;          // index of end of cell
	

	// GPU data
	float *d_pos_;        // these are the CUDA deviceMem Pos
	float *d_velo_;
	float *d_velo_delta_;
	uint *d_type_;

	// grid data for sorting method
	uint  *d_hash_; // grid hash value for each particle
	uint  *d_index_sorted_;// particle index for each particle
	uint  *d_cell_start_;        // index of start of each cell in sorted list
	uint  *d_cell_end_;          // index of end of cell

	uint   m_gridSortBits;

	SimulationEnv env_;
	SimulationSphereProto protos_;
	uint3 grid_size_;
	uint3 grid_exp_;
	uint cell_num_;

	glm::vec3 origin_;
	glm::vec3 room_size_;

};

#endif // !PHYSICSENGINE_H
