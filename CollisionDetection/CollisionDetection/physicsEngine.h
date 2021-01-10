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

// three initialization modes of spheres
enum InitMode {
	SPREAD_MODE,
	CUBE_MODE,
	RANDOM_MODE
};

class PhysicsEngine {
public:
	PhysicsEngine(uint sphere_num, 
		bool gpu_mode,
		glm::vec3 origin, 
		glm::vec3 room_size, 
		uint hash_block=HASH_BLOCK, 
		bool brutal_mode = BRUTAL_MODE,
		InitMode init_mode= CUBE_MODE);

	~PhysicsEngine();

	// output updated sphere positions
	float* outputPos();

	// simulate the collision and motion of all spheres in given amount of time (elapse) 
	void update(float elapse);

	// get the sphere types for rendering
	uint *getSphereType() { return h_type_; }

protected:
	// initialize the CPU and GPU memory
	void initMemory();

	// initialize the environment
	void initEnvironment();

	// initialize the type, position and velocity of all spheres
	void initSpheres();

	// release the CPU and GPU memory
	void releaseMemory();

	// generate jitters with the given magnitude
	float genJitter(float magnitude);

protected:
	// use GPU for calculation or not
	bool gpu_mode_;

	// if use CPU, whether use brutal method
	bool brutal_mode_;
	
	// number of spheres
	uint sphere_num_;

	// CPU data
	float *h_pos_; // positions
	float *h_velo_;  // velocities
	float *h_accel_; // accelerations
	uint *h_type_; // sphere types

	uint  *h_hash_; // hash values
	uint  *h_index_sorted_; // sorted indices
	uint  *h_cell_start_; // sphere index for the start of each cell
	uint  *h_cell_end_; // sphere index for the end of each cell
	
	// GPU data
	float *d_pos_;
	float *d_velo_;
	float *d_accel_;
	uint *d_type_;

	uint  *d_hash_; 
	uint  *d_index_sorted_;
	uint  *d_cell_start_;
	uint  *d_cell_end_;

	// simulation environment
	SimulationEnv env_;
	SimulationSphereProto protos_;
	InitMode init_mode_; // sphere position initialization mode

	// values for hashing and dividing space
	uint hash_block_;
	uint max_hash_value_;
	float cell_size_;

	// room origin and size
	glm::vec3 origin_;
	glm::vec3 room_size_;
};

#endif // !PHYSICSENGINE_H
