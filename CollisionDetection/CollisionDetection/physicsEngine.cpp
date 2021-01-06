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
	m_params.grid_exp = grid_exp_;
	// grid_size = 20;
	m_gridSize.x = m_gridSize.y = m_gridSize.z = grid_size;

	cell_num_ = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	//    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 24;    // increase this for larger grids
	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.cell_num = cell_num_;
	m_params.sphere_num = sphere_num;

	m_params.max_radius = 1.0f / 20.0f;
	//m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	//m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
	float cellSize = m_params.max_radius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	m_params.globalDamping = 1.0f;

	m_params.e = 0.5f;

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
	/*for (uint i = 0; i < sphere_num; i++) 
	{
		h_mass_[i] = 1.0f;
		h_radius_[i] = m_params.max_radius;
	}
	printf("%f, %f", h_mass_[0], h_radius_[0]);*/
	memset(h_rest_, 0, space_1x);

	// allocate GPU data
	allocateArray((void **)&d_pos_, space_3x);
	allocateArray((void **)&d_velo_, space_3x);
	allocateArray((void **)&d_velo_delta_, space_3x);
	allocateArray((void **)&d_mass_, space_1x);
	allocateArray((void **)&d_rest_, space_1x);
	allocateArray((void **)&d_radius_, space_1x);

	allocateArray((void **)&d_pos_sorted_, space_3x);
	allocateArray((void **)&d_velo_sorted_, space_3x);

	allocateArray((void **)&d_hash_, sphere_num * sizeof(uint));
	allocateArray((void **)&d_index_, sphere_num * sizeof(uint));

	allocateArray((void **)&m_dCellStart, cell_num_ * sizeof(uint));
	allocateArray((void **)&m_dCellEnd, cell_num_ * sizeof(uint));
	dSetupSimulation(&m_params);
	initData();
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
	freeArray(d_pos_sorted_);
	freeArray(d_velo_sorted_);

	freeArray(d_hash_);
	freeArray(d_index_);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);
}

void PhysicsEngine::initData()
{
	float jitter = m_params.max_radius*0.01f;
	uint s = (int)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
	uint gridSize[3];
	gridSize[0] = gridSize[1] = gridSize[2] = s;
	srand(1973);

	for (uint z = 0; z < gridSize[2]; z++)
	{
		for (uint y = 0; y < gridSize[1]; y++)
		{
			for (uint x = 0; x < gridSize[0]; x++)
			{
				uint i = (z*gridSize[1] * gridSize[0]) + (y*gridSize[0]) + x;

				if (i < sphere_num_)
				{
					h_pos_[i * 3] = (m_params.max_radius*2.0f * x) + m_params.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 3 + 1] = (m_params.max_radius*2.0f * y) + m_params.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 3 + 2] = (m_params.max_radius*2.0f * z) + m_params.max_radius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					//h_pos_[i * 3 + 3] = 1.0f;

					h_velo_[i * 3] = 0.0f;
					h_velo_[i * 3 + 1] = 0.0f;
					h_velo_[i * 3 + 2] = 0.0f;
					//h_velo_[i * 3 + 3] = 0.0f;

					uint prototype_id = rand() % PROTOTYPE_NUM;
					Sphere prototype = PROTOTYPES[prototype_id];
					h_color_[i * 3] = prototype.r_;
					h_color_[i * 3 + 1] = prototype.g_;
					h_color_[i * 3 + 2] = prototype.b_;
					h_mass_[i] = prototype.mass_;
					h_radius_[i] = prototype.radius_;
					h_rest_[i] = prototype.rest_;


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
	//threadSync();
	copyArrayFromDevice(h_pos_, d_pos_, sizeof(float) * 3 *sphere_num_);
	// checkCudaErrors(cudaMemcpy(h_pos_, d_pos_, sizeof(float) * 4 * sphere_num_, cudaMemcpyDeviceToHost));
	return h_pos_;
}

void PhysicsEngine::update(float elapse)
{
	
	// update constants
	dSetupSimulation(&m_params);

	// integrate
	dUpdateDynamics(
		d_pos_,
		d_velo_,
		d_velo_delta_,
		d_radius_,
		elapse,
		sphere_num_);

	// calculate grid hash
	dHashifyAndSort(
		d_hash_,
		d_index_,
		d_pos_,
		sphere_num_);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	dCollectCells(
		m_dCellStart,
		m_dCellEnd,
		d_hash_,
		sphere_num_, 
		cell_num_);

	// process collisions
	collide(
		d_velo_delta_,
		d_pos_,
		d_velo_,
		d_radius_,
		d_mass_,
		d_index_,
		m_dCellStart,
		m_dCellEnd,
		sphere_num_,
		cell_num_);

}

inline void PhysicsEngine::createArrayOnGPU()
{
}
