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
m_hPos(0),
m_hVel(0),
pos_arr(0),
vel_arr(0),
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

	// allocate host storage
	m_hPos = new float[sphere_num * 4];
	m_hVel = new float[sphere_num * 4];
	memset(m_hPos, 0, sphere_num * 4 * sizeof(float));
	memset(m_hVel, 0, sphere_num * 4 * sizeof(float));

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

	// allocate GPU data
	unsigned int memSize = sizeof(float) * 4 * sphere_num;

	allocateArray((void **)&m_cudaPosVBO, memSize);

	allocateArray((void **)&vel_arr, memSize);

	allocateArray((void **)&m_dSortedPos, memSize);
	allocateArray((void **)&m_dSortedVel, memSize);

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
	delete[] m_hPos;
	delete[] m_hVel;
	delete[] m_hCellStart;
	delete[] m_hCellEnd;

	freeArray(vel_arr);
	freeArray(m_dSortedPos);
	freeArray(m_dSortedVel);

	freeArray(m_dGridParticleHash);
	freeArray(m_dGridParticleIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);

	freeArray(m_cudaPosVBO);
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
					m_hPos[i * 4] = (m_params.particleRadius*2.0f * x) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					m_hPos[i * 4 + 1] = (m_params.particleRadius*2.0f * y) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					m_hPos[i * 4 + 2] = (m_params.particleRadius*2.0f * z) + m_params.particleRadius - 1.0f + (rand() / (float)RAND_MAX*2.0f - 1.0f)*jitter;
					m_hPos[i * 4 + 3] = 1.0f;

					m_hVel[i * 4] = 0.0f;
					m_hVel[i * 4 + 1] = 0.0f;
					m_hVel[i * 4 + 2] = 0.0f;
					m_hVel[i * 4 + 3] = 0.0f;
				}
			}
		}
	}
	copyArrayToDevice(m_cudaPosVBO, m_hPos,0, sphere_num_ * 4 * sizeof(float));
	copyArrayToDevice(vel_arr, m_hVel, 0, sphere_num_ * 4 * sizeof(float));
}

float * PhysicsEngine::outputPos()
{
	//threadSync();
	copyArrayFromDevice_(m_hPos, m_cudaPosVBO, sizeof(float) * 4 *sphere_num_);
	// checkCudaErrors(cudaMemcpy(m_hPos, m_cudaPosVBO, sizeof(float) * 4 * sphere_num_, cudaMemcpyDeviceToHost));
	return m_hPos;
}

void PhysicsEngine::update(float deltaTime)
{
	
	float *dPos = (float *) m_cudaPosVBO;

	// update constants
	setParameters(&m_params);

	// integrate
	integrateSystem(
		dPos,
		vel_arr,
		deltaTime,
		sphere_num_);

	// calculate grid hash
	calcHash(
		m_dGridParticleHash,
		m_dGridParticleIndex,
		dPos,
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
		dPos,
		vel_arr,
		sphere_num_,
		m_numGridCells);

	// process collisions
	collide(
		vel_arr,
		m_dSortedPos,
		m_dSortedVel,
		m_dGridParticleIndex,
		m_dCellStart,
		m_dCellEnd,
		sphere_num_,
		m_numGridCells);

}

inline void PhysicsEngine::createArrayOnGPU()
{
}
