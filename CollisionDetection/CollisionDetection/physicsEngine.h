#ifndef PHYSICSENGINE_H
#define PHYSICSENGINE_H

# include "global.h"
#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

class PhysicsEngine
{
public:
	PhysicsEngine(uint sphere_num= SPHERE_NUM, uint grid_size=GRID_SIZE, bool gpu_mode= GPU_MODE);
	~PhysicsEngine();
	void initRenderer();
	float* outputPos();
	void update(float deltaTime);

protected:
	inline void createArrayOnGPU();

protected:
	bool gpu_mode_;

	uint sphere_num_;
	// CPU data
	float *h_pos_;              // particle positions
	float *h_velo_;              // particle velocities
	float *h_velo_delta_;

	float *h_mass_;
	float *h_rest_;
	float *h_radius_;
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
	float *d_mass_;
	float *d_rest_;
	float *d_radius_;
	uint *d_type_;

	// grid data for sorting method
	uint  *d_hash_; // grid hash value for each particle
	uint  *d_index_sorted_;// particle index for each particle
	uint  *d_cell_start_;        // index of start of each cell in sorted list
	uint  *d_cell_end_;          // index of end of cell

	uint   m_gridSortBits;

	SimulationEnv env_;
	SimulationSphereStats stats_;
	uint3 m_gridSize;
	uint3 grid_exp_;
	uint cell_num_;


public:

	void setEnvDrag(float x)
	{
		env_.drag = x;
	}
	void setEnvGravity(float x)
	{
		env_.gravity = make_float3(0.0f, x, 0.0f);
	}

	void setEnvStiffness(float x)
	{
		env_.stiffness = x;
	}

	void setEnvDamping(float val)
	{
		env_.damping = val;
	}

	void setBoundaryDamping(float val)
	{
		env_.boundary_damping = val;
	}

	void setEnvFriction(float val)
	{
		env_.friction = val;
	}

	uint *getSphereType()
	{
		return h_type_;
	}
};

#endif // !PHYSICSENGINE_H
