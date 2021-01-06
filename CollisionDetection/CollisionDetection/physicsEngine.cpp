#include "physicsEngine.h"
#include "physicsEngine.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "sphere.h"

PhysicsEngine::PhysicsEngine(uint sphere_num, uint grid_size): sphere_num_(sphere_num),
h_pos_(0),
h_velo_(0),
d_pos_(0),
d_velo_(0)
{
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

	env_.max_radius = 1.0f / 24.0f;

	env_.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	float cell_size = env_.max_radius * 2.0f;  // cell size equal to particle diameter
	env_.cell_size = make_float3(cell_size, cell_size, cell_size);

	float damping = 0.999f;
	float gravity = 0.05f;//0.001f;
	float collide_spring = 2.5f;//0.5f;
	float collide_damping = 0.02f;
	float collide_shear = 0.1f;
	float collide_e = 0.2f;
	float boundary_damping = -0.5f;
	setDrag(damping);
	setGravity(-gravity);
	setBoundaryDamping(boundary_damping);
	setCollideSpring(collide_spring);
	setCollideDamping(collide_damping);
	setCollideShear(collide_shear);
	setCollideE(collide_e);


	// allocate host storage
	uint space_1x = sizeof(float) * sphere_num;
	uint space_3x = space_1x * 3;
	uint space_4x = space_1x * 4;
	
	h_pos_ = new float[sphere_num * 3];
	h_velo_ = new float[sphere_num * 3];
	h_color_ = new float[sphere_num * 3];
	h_mass_ = new float[sphere_num];
	h_radius_ = new float[sphere_num];
	h_rest_ = new float[sphere_num];
	memset(h_pos_, 0, space_3x);
	memset(h_velo_, 0, space_3x);
	memset(h_color_, 0, space_3x);
	memset(h_rest_, 0, space_1x);

	// allocate GPU data
	allocateArray((void **)&d_pos_, space_3x);
	allocateArray((void **)&d_velo_, space_3x);
	allocateArray((void **)&d_velo_delta_, space_3x);
	zeroizeArray(d_velo_delta_, space_3x);
	allocateArray((void **)&d_mass_, space_1x);
	allocateArray((void **)&d_rest_, space_1x);
	allocateArray((void **)&d_radius_, space_1x);

	// parameters for middle calculations on GPU
	allocateArray((void **)&d_hash_, sphere_num * sizeof(uint));
	allocateArray((void **)&d_index_sorted_, sphere_num * sizeof(uint));
	allocateArray((void **)&d_cell_start_, cell_num_ * sizeof(uint));
	allocateArray((void **)&d_cell_end_, cell_num_ * sizeof(uint));

	initData();
	dSetupSimulation(&env_);
	threadSync();
}


PhysicsEngine::~PhysicsEngine()
{
	delete[] h_pos_;
	delete[] h_velo_; 
	delete[] h_color_;
	delete[] h_mass_;
	delete[] h_radius_;
	delete[] h_rest_;

	freeArray(d_velo_);
	freeArray(d_velo_delta_);
	freeArray(d_pos_);
	freeArray(d_mass_);
	freeArray(d_radius_);
	freeArray(d_rest_);

	freeArray(d_hash_);
	freeArray(d_index_sorted_);
	freeArray(d_cell_start_);
	freeArray(d_cell_end_);
}

void PhysicsEngine::initData()
{
	float jitter = env_.max_radius*0.01f;
	uint s = (int)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
	uint grid_size[3];
	grid_size[0] = grid_size[1] = grid_size[2] = s;
	srand(1973);

	for (uint z = 0; z < grid_size[2]; z++)
	{
		for (uint y = 0; y < grid_size[1]; y++)
		{
			for (uint x = 0; x < grid_size[0]; x++)
			{
				uint i = (z*grid_size[1] * grid_size[0]) + (y*grid_size[0]) + x;

				if (i < sphere_num_)
				{
					h_pos_[i * 3] = (env_.max_radius*2.0f * x) + env_.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 3 + 1] = (env_.max_radius*2.0f * y) + env_.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 3 + 2] = (env_.max_radius*2.0f * z) + env_.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;

					h_velo_[i * 3] = 0.0f;
					h_velo_[i * 3 + 1] = 0.0f;
					h_velo_[i * 3 + 2] = 0.0f;

					uint prototype_id = rand() % PROTOTYPE_NUM;
					Sphere prototype = PROTOTYPES[prototype_id];
					h_color_[i * 3] = prototype.r;
					h_color_[i * 3 + 1] = prototype.g;
					h_color_[i * 3 + 2] = prototype.b;
					h_mass_[i] = prototype.mass;
					h_radius_[i] = prototype.radius;
					h_rest_[i] = prototype.rest;
				}
			}
		}
	}
	copyArrayToDevice(d_pos_, h_pos_,0, sphere_num_ * 3 * sizeof(float));
	copyArrayToDevice(d_velo_, h_velo_, 0, sphere_num_ * 3 * sizeof(float));
	copyArrayToDevice(d_radius_, h_radius_, 0, sphere_num_ * sizeof(float));
	copyArrayToDevice(d_mass_, h_mass_, 0, sphere_num_ * sizeof(float));
}

float * PhysicsEngine::outputPos()
{
	copyArrayFromDevice(h_pos_, d_pos_, sizeof(float) * 3 *sphere_num_);
	return h_pos_;
}

void PhysicsEngine::update(float elapse)
{
	// dSetupSimulation(&env_);

	dUpdateDynamics(
		d_pos_,
		d_velo_,
		d_velo_delta_,
		d_radius_,
		elapse,
		sphere_num_);

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
		d_radius_,
		d_mass_,
		d_rest_,
		d_index_sorted_,
		d_cell_start_,
		d_cell_end_,
		sphere_num_);
}

inline void PhysicsEngine::createArrayOnGPU()
{
}
