#include "physicsEngine.h"
#include "physicsEngine.cuh"
#include "particleSystem.h"
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

PhysicsEngine::PhysicsEngine(uint sphere_num, uint grid_size): sphere_num_(sphere_num),
h_pos_(0),
h_velo_(0),
d_pos_(0),
d_velo_(0),
m_solverIterations(1)
{
	m_gridSize.x = m_gridSize.y = m_gridSize.z = grid_size;

	m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	//    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids
	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = sphere_num;

	m_params.particleRadius = 1.0f / 64.0f;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	//    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
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
	
	h_pos_ = new float[sphere_num * 4];
	h_velo_ = new float[sphere_num * 4];
	h_color_ = new float[sphere_num * 3];
	h_mass_ = new float[sphere_num];
	h_radius_ = new float[sphere_num];
	h_rest_ = new float[sphere_num];
	memset(h_pos_, 0, space_4x);
	memset(h_velo_, 0, space_4x);
	memset(h_color_, 0, space_3x);
	memset(h_mass_, 0, space_1x);
	memset(h_radius_, 0, space_1x);
	memset(h_rest_, 0, space_1x);

	// allocate GPU data
	allocateArray((void **)&d_pos_, space_4x);
	allocateArray((void **)&d_velo_, space_4x);
	allocateArray((void **)&d_mass_, space_1x);
	allocateArray((void **)&d_rest_, space_1x);
	allocateArray((void **)&d_radius_, space_1x);

	allocateArray((void **)&m_dSortedPos, space_4x);
	allocateArray((void **)&m_dSortedVel, space_4x);

	allocateArray((void **)&m_dGridParticleHash, sphere_num * sizeof(uint));
	allocateArray((void **)&m_dGridParticleIndex, sphere_num * sizeof(uint));

	allocateArray((void **)&m_dCellStart, m_numGridCells * sizeof(uint));
	allocateArray((void **)&m_dCellEnd, m_numGridCells * sizeof(uint));
	setParameters(&m_params);
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
	freeArray(d_pos_);
	freeArray(d_mass_);
	freeArray(d_radius_);
	freeArray(d_rest_);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);
}

void PhysicsEngine::initData()
{
	float jitter = m_params.particleRadius*0.01f;
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
					h_pos_[i * 4] = (m_params.particleRadius*2.0f * x) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 4 + 1] = (m_params.particleRadius*2.0f * y) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 4 + 2] = (m_params.particleRadius*2.0f * z) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					h_pos_[i * 4 + 3] = 1.0f;

					h_velo_[i * 4] = 0.0f;
					h_velo_[i * 4 + 1] = 0.0f;
					h_velo_[i * 4 + 2] = 0.0f;
					h_velo_[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
	copyArrayToDevice(d_pos_, h_pos_,0, sphere_num_ * 4 * sizeof(float));
	copyArrayToDevice(d_velo_, h_velo_, 0, sphere_num_ * 4 * sizeof(float));
}

float * PhysicsEngine::outputPos()
{
	//threadSync();
	copyArrayFromDevice_(h_pos_, d_pos_, sizeof(float) * 4 *sphere_num_);
	// checkCudaErrors(cudaMemcpy(h_pos_, d_pos_, sizeof(float) * 4 * sphere_num_, cudaMemcpyDeviceToHost));
	return h_pos_;
}

void PhysicsEngine::update(float deltaTime)
{
	
	// update constants
	setParameters(&m_params);

	// integrate
	integrateSystem(
		d_pos_,
		d_velo_,
		deltaTime,
		sphere_num_);

	// calculate grid hash
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		d_pos_,
		sphere_num_);

	// sort particles based on hash
	sortParticles(m_dGridParticleHash, m_dGridParticleIndex, sphere_num_);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
		m_dCellStart,
		m_dCellEnd,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleHash,
		m_dGridParticleIndex,
		d_pos_,
		d_velo_,
		sphere_num_,
		m_numGridCells);

	// process collisions
	collide(
		d_velo_,
		m_dSortedPos,
		m_dSortedVel,
		d_radius_,
		d_mass_,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		sphere_num_,
		m_numGridCells);

}

inline void PhysicsEngine::createArrayOnGPU()
{
}
