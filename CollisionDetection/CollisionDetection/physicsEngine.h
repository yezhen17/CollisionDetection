#ifndef PHYSICSENGINE_H
#define PHYSICSENGINE_H

# include "global.h"
#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

class PhysicsEngine
{
public:
	PhysicsEngine(uint sphere_num= SPHERE_NUM, uint grid_size=GRID_SIZE);
	~PhysicsEngine();
	void initData();
	float* outputPos();
	void update(float deltaTime);

protected:
	inline void createArrayOnGPU();

protected:
	uint sphere_num_;
	// CPU data
	float *m_hPos;              // particle positions
	float *m_hVel;              // particle velocities

	uint  *m_hParticleHash;
	uint  *m_hCellStart;
	uint  *m_hCellEnd;

	// GPU data
	float *pos_arr;
	float *vel_arr;

	float *m_dSortedPos;
	float *m_dSortedVel;

	// grid data for sorting method
	uint  *m_dGridParticleHash; // grid hash value for each particle
	uint  *m_dGridParticleIndex;// particle index for each particle
	uint  *m_dCellStart;        // index of start of each cell in sorted list
	uint  *m_dCellEnd;          // index of end of cell

	

	uint   m_gridSortBits;

	//uint   m_posVbo;            // vertex buffer object for particle positions
	//uint   m_colorVBO;          // vertex buffer object for colors

	float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos
	//float *m_cudaColorVBO;      // these are the CUDA deviceMem Color

	//struct cudaGraphicsResource *m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	//struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	// params
	SimParams m_params;
	uint3 m_gridSize;
	uint m_numGridCells;

	uint m_solverIterations;

public:
	void setIterations(int i)
	{
		m_solverIterations = i;
	}

	void setDamping(float x)
	{
		m_params.globalDamping = x;
	}
	void setGravity(float x)
	{
		m_params.gravity = make_float3(0.0f, x, 0.0f);
	}

	void setCollideSpring(float x)
	{
		m_params.spring = x;
	}
	void setCollideDamping(float x)
	{
		m_params.damping = x;
	}
	void setCollideShear(float x)
	{
		m_params.shear = x;
	}
	void setCollideAttraction(float x)
	{
		m_params.attraction = x;
	}

	void setColliderPos(float3 x)
	{
		m_params.colliderPos = x;
	}
};

#endif // !PHYSICSENGINE_H