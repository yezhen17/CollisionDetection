/*
 * Implementation of the physics engine
 */

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "physicsEngine.h"
#include "dSimulation.cuh"
#include "hSimulation.h"
#include "environment.h"
#include "sphere.h"


PhysicsEngine::PhysicsEngine(uint sphere_num, glm::vec3 origin, glm::vec3 room_size, uint grid_size, bool gpu_mode):
	sphere_num_(sphere_num),
	gpu_mode_(gpu_mode),
	h_pos_(0),
	h_velo_(0),
	d_pos_(0),
	d_velo_(0) {
	grid_exp_.x = grid_exp_.y = grid_exp_.z = ceil(log2(grid_size));
	printf("%d", grid_exp_.x);
	env_.grid_exp = grid_exp_;
	// grid_size = 20;
	m_gridSize.x = m_gridSize.y = m_gridSize.z = grid_size;

	cell_num_ = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	//    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 24;    // increase this for larger grids
	// set simulation parameters
	env_.grid_size = m_gridSize;
	env_.cell_num = cell_num_;
	env_.sphere_num = sphere_num;

	env_.max_radius = 1.0f / 32.0f;

	float cell_size = env_.max_radius * 2.0f;  // cell size equal to particle diameter
	env_.cell_size = make_float3(cell_size, cell_size, cell_size);

	env_.min_corner = make_float3(origin.x, origin.y, origin.z);
	env_.max_corner = env_.min_corner + make_float3(room_size.x, room_size.y, room_size.z);

	// initialize simulation environment
	float drag = 0.999f;
	float gravity = 0.05f;//0.001f;
	float stiffness = 2.5f;//0.5f;
	float damping = 0.2f;
	float friction = 0.0f;
	float boundary_damping = -0.5f;
	setEnvDrag(drag);
	setEnvGravity(-gravity);
	setBoundaryDamping(boundary_damping);
	setEnvStiffness(stiffness);
	setEnvDamping(damping);
	setEnvFriction(friction);

	for (uint i = 0; i < PROTOTYPE_NUM; ++i) {
		Sphere prototype = PROTOTYPES[i];
		protos_.masses[i] = prototype.mass;
		protos_.radii[i] = prototype.radius;
	}

	// allocate CPU memory
	uint space_1xf = sizeof(float) * sphere_num;
	uint space_3xf = space_1xf * 3;
	uint space_1xu = sizeof(uint) * sphere_num;
	
	h_pos_ = new float[sphere_num * 3];
	h_velo_ = new float[sphere_num * 3];
	h_velo_delta_ = new float[sphere_num * 3];
	h_type_ = new uint[sphere_num];

	// allocate parameters for middle calculations in CPU memory
	h_hash_ = new uint[sphere_num];
	h_index_sorted_ = new uint[sphere_num];
	h_cell_start_ = new uint[cell_num_];
	h_cell_end_ = new uint[cell_num_];

	memset(h_pos_, 0, space_3xf);
	memset(h_velo_, 0, space_3xf);
	memset(h_velo_delta_, 0, space_3xf);
	memset(h_type_, 0, space_1xu);
	memset(h_hash_, 0, space_1xf);
	memset(h_index_sorted_, 0, space_1xf);
	memset(h_cell_start_, 0, cell_num_ * sizeof(uint));
	memset(h_cell_end_, 0, cell_num_ * sizeof(uint));

	initSpheres();
	hSetupSimulation(&env_, &protos_);

	if (gpu_mode_) {
		// allocate GPU data
		allocateArray((void **)&d_pos_, space_3xf);
		allocateArray((void **)&d_velo_, space_3xf);
		allocateArray((void **)&d_velo_delta_, space_3xf);
		allocateArray((void **)&d_type_, space_1xu);

		// allocate parameters for middle calculations on GPU
		allocateArray((void **)&d_hash_, space_1xu);
		allocateArray((void **)&d_index_sorted_, space_1xu);
		allocateArray((void **)&d_cell_start_, cell_num_ * sizeof(uint));
		allocateArray((void **)&d_cell_end_, cell_num_ * sizeof(uint));

		// copy these initialized values to GPU
		copyArrayToDevice(d_pos_, h_pos_, 0, sphere_num_ * 3 * sizeof(float));
		copyArrayToDevice(d_velo_, h_velo_, 0, sphere_num_ * 3 * sizeof(float));
		copyArrayToDevice(d_type_, h_type_, 0, sphere_num_ * sizeof(uint));

		dSetupSimulation(&env_, &protos_);
	}
}

PhysicsEngine::~PhysicsEngine() {
	// release all memory, CPU or GPU
	delete[] h_pos_;
	delete[] h_velo_; 
	delete[] h_velo_delta_;
	delete[] h_type_;
	delete[] h_hash_;
	delete[] h_index_sorted_;
	delete[] h_cell_start_;
	delete[] h_cell_end_;

	if (gpu_mode_) {
		freeArray(d_velo_);
		freeArray(d_velo_delta_);
		freeArray(d_pos_);
		freeArray(d_type_);
		freeArray(d_hash_);
		freeArray(d_index_sorted_);
		freeArray(d_cell_start_);
		freeArray(d_cell_end_);
	}
}

void PhysicsEngine::initSpheres() {
	float jitter = env_.max_radius*0.01f;
	uint s = (int)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
	uint grid_size[3];
	grid_size[0] = grid_size[1] = grid_size[2] = s;
	srand(1973);

	for (uint z = 0; z < grid_size[2]; z++) {
		for (uint y = 0; y < grid_size[1]; y++) {
			for (uint x = 0; x < grid_size[0]; x++) {
				uint i = (z*grid_size[1] * grid_size[0]) + (y*grid_size[0]) + x;

				if (i < sphere_num_) {
					h_pos_[i * 3] = (env_.max_radius*2.0f * x + env_.max_radius + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter);
					h_pos_[i * 3 + 1] = (env_.max_radius*2.0f * y + env_.max_radius + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter);
					h_pos_[i * 3 + 2] = (env_.max_radius*2.0f * z + env_.max_radius + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter);
					h_velo_[i * 3] = 0.0f;
					h_velo_[i * 3 + 1] = 0.0f;
					h_velo_[i * 3 + 2] = 0.0f;

					uint proto_id = rand() % PROTOTYPE_NUM;
					h_type_[i] = proto_id;
				}
			}
		}
	}
}

float * PhysicsEngine::outputPos() {
	// if GPU mode, first copy the data from device to host
	if (gpu_mode_) {
		copyArrayFromDevice(h_pos_, d_pos_, sizeof(float) * 3 * sphere_num_);
	}
	return h_pos_;
}

void PhysicsEngine::update(float elapse) {
	// choose CPU or GPU to do the calculation
	if (gpu_mode_) {
		dHashifyAndSort(
			d_hash_,
			d_index_sorted_,
			d_pos_,
			sphere_num_);

		dCollectCells(
			d_cell_start_,
			d_cell_end_,
			d_hash_,
			sphere_num_,
			cell_num_);

		dNarrowPhaseCollisionDetection(
			d_velo_delta_,
			d_pos_,
			d_velo_,
			d_type_,
			d_index_sorted_,
			d_cell_start_,
			d_cell_end_,
			sphere_num_);

		dUpdateDynamics(
			d_pos_,
			d_velo_,
			d_velo_delta_,
			d_type_,
			elapse,
			sphere_num_);
	} else {
		hHashifyAndSort(
			h_hash_,
			h_index_sorted_,
			(float3 *)h_pos_,
			sphere_num_);

		hCollectCells(
			h_cell_start_,
			h_cell_end_,
			h_hash_,
			sphere_num_,
			cell_num_);

		hNarrowPhaseCollisionDetection(
			(float3 *)h_velo_delta_,
			(float3 *)h_pos_,
			(float3 *)h_velo_,
			h_type_,
			h_index_sorted_,
			h_cell_start_,
			h_cell_end_,
			sphere_num_);

		hUpdateDynamics(
			(float3 *)h_pos_,
			(float3 *)h_velo_,
			(float3 *)h_velo_delta_,
			h_type_,
			elapse,
			sphere_num_);
	}
}
