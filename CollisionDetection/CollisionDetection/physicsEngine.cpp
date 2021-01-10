/*
 * Implementation of the physics engine
 */

#include "physicsEngine.h"
#include "dSimulation.cuh"
#include "hSimulation.h"
#include "environment.h"
#include "sphere.h"


PhysicsEngine::PhysicsEngine(uint sphere_num, bool gpu_mode, glm::vec3 origin, glm::vec3 room_size,
	uint hash_block, bool brutal_mode, InitMode init_mode):
	sphere_num_(sphere_num),
	gpu_mode_(gpu_mode),
	brutal_mode_(brutal_mode),
	origin_(origin),
	room_size_(room_size),
	hash_block_(hash_block),
	max_hash_value_(hash_block * hash_block * hash_block),
    init_mode_(init_mode) {
	initMemory();
}

PhysicsEngine::~PhysicsEngine() {
	releaseMemory();
}

void PhysicsEngine::initMemory() {
	// allocate CPU memory
	uint space_1xf = sizeof(float) * sphere_num_;
	uint space_3xf = space_1xf * 3;
	uint space_1xu = sizeof(uint) * sphere_num_;

	h_pos_ = new float[sphere_num_ * 3];
	h_velo_ = new float[sphere_num_ * 3];
	h_type_ = new uint[sphere_num_];
	
	memset(h_pos_, 0, space_3xf);
	memset(h_velo_, 0, space_3xf);
	memset(h_type_, 0, space_1xu);
	
	initEnvironment();
	initSpheres();
	
	if (gpu_mode_) {
		// allocate GPU data
		allocateArray((void **)&d_pos_, space_3xf);
		allocateArray((void **)&d_velo_, space_3xf);
		allocateArray((void **)&d_accel_, space_3xf);
		allocateArray((void **)&d_type_, space_1xu);
		zeroizeArray(d_accel_, space_3xf);

		// allocate parameters for middle calculations on GPU
		allocateArray((void **)&d_hash_, space_1xu);
		allocateArray((void **)&d_index_sorted_, space_1xu);
		allocateArray((void **)&d_cell_start_, max_hash_value_ * sizeof(uint));
		allocateArray((void **)&d_cell_end_, max_hash_value_ * sizeof(uint));

		// copy these initialized values to GPU
		copyHost2Device(d_pos_, h_pos_, 0, space_3xf);
		copyHost2Device(d_velo_, h_velo_, 0, space_3xf);
		copyHost2Device(d_type_, h_type_, 0, space_1xu);

		dSetupSimulation(&env_, &protos_);
	} 
	else {
		// allocate parameters for middle calculations in CPU memory
		h_accel_ = new float[sphere_num_ * 3];
		h_hash_ = new uint[sphere_num_];
		h_index_sorted_ = new uint[sphere_num_];
		h_cell_start_ = new uint[max_hash_value_];
		h_cell_end_ = new uint[max_hash_value_];

		memset(h_accel_, 0, space_3xf);
		memset(h_hash_, 0, space_1xf);
		memset(h_index_sorted_, 0, space_1xf);
		memset(h_cell_start_, 0, max_hash_value_ * sizeof(uint));
		memset(h_cell_end_, 0, max_hash_value_ * sizeof(uint));
		
		hSetupSimulation(&env_, &protos_);
	}
}

void PhysicsEngine::initEnvironment() {
	env_.sphere_num = sphere_num_;
	env_.max_hash_value = max_hash_value_;

	float max_radius = 0.0f;
	float min_mass = 999.9f;
	for (uint i = 0; i < PROTOTYPE_NUM; ++i) {
		Sphere prototype = PROTOTYPES[i];
		if (prototype.radius > max_radius) {
			max_radius = prototype.radius;
		}
		if (prototype.mass < min_mass) {
			min_mass = prototype.mass;
		}
		protos_.masses[i] = prototype.mass;
		protos_.radii[i] = prototype.radius;
		for (uint j = 0; j < PROTOTYPE_NUM; ++j) {
			float rest = RESTITUTION[i][j];
			float ln_rest = logf(rest);
			float denom = sqrtf(ln_rest * ln_rest + PI * PI);
			float numer = 2 * ln_rest;
			protos_.damping[i][j] = -numer / denom;
			protos_.restitution[i][j] = rest;
			// std::cout << rest << ", " << -numer / denom << std::endl;
		}
	}
	if (sphere_num_ > 4096) {
		max_radius = PROTOTYPES[3].radius;
		min_mass = PROTOTYPES[3].mass;
	}

	// cell size is equal to the maximum sphere radius * 2
	cell_size_ = max_radius * 2.0f;  
	env_.cell_size = cell_size_;

	// the two extreme corners of the room
	env_.min_corner = make_float3(origin_.x, origin_.y, origin_.z);
	env_.max_corner = env_.min_corner + make_float3(room_size_.x, room_size_.y, room_size_.z);

	// environment parameters 
	env_.drag = DRAG;
	env_.gravity = make_float3(0.0f, -GRAVITY, 0.0f);
	env_.stiffness = STIFFNESS * min_mass;
	env_.damping = DAMPING * min_mass;
	env_.friction = FRICTION * min_mass;
}

void PhysicsEngine::initSpheres() {
	srand(2021);
	
	if (init_mode_ == SPREAD_MODE || init_mode_ == CUBE_MODE) {
		float half_cell_size = cell_size_ * 0.5f;
		float jitter_magnitude = half_cell_size * 0.02f;
		float velo_magnitude = half_cell_size * 0.02f;
		uint x_num, y_num, z_num;
		
		// spread mode means that: starting from the top
		// the spheres spread the whole x-z surface of that height constrained by the boundary
		if (init_mode_ == SPREAD_MODE) {
			x_num = (uint)ceilf(room_size_.x / cell_size_);
			z_num = (uint)ceilf(room_size_.z / cell_size_);
			y_num = (uint)ceilf((float)sphere_num_ / x_num / z_num);
		}
		// cube mode means that: starting from the top corner
		// the spheres form a cube (as "cubic" as possible)
		else {
			x_num = (uint)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
			y_num = (uint)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
			z_num = (uint)ceilf(powf((float)sphere_num_, 1.0f / 3.0f));
		}
		
		for (uint y = 0; y < y_num; y++) {
			for (uint z = 0; z < z_num; z++) {
				for (uint x = 0; x < x_num; x++) {
					uint index = (y * z_num + z) * x_num + x;
					if (index < sphere_num_) {
						h_pos_[index * 3] = origin_.x + cell_size_ * x + half_cell_size + genJitter(jitter_magnitude);
						h_pos_[index * 3 + 1] = origin_.y + room_size_.y - cell_size_ * (y_num - 1 - y) - half_cell_size + genJitter(jitter_magnitude);
						h_pos_[index * 3 + 2] = origin_.z + cell_size_ * z + half_cell_size + genJitter(jitter_magnitude);
						h_velo_[index * 3] = genJitter(velo_magnitude);
						h_velo_[index * 3 + 1] = genJitter(velo_magnitude);
						h_velo_[index * 3 + 2] = genJitter(velo_magnitude);
						if (sphere_num_ > 4096) {
							h_type_[index] = 3;
						}
						else {
							uint proto_id = rand() % PROTOTYPE_NUM;
							h_type_[index] = proto_id;
						}
					}
				}
			}
		}
	}
	// random mode means that: split the room into blocks (size = 2*cell size)
	// and put spheres randomly into blocks
	else {
		float block_size = 2 * cell_size_;
		uint block_num_x = (uint)floor(room_size_.x / block_size);
		uint block_num_y = (uint)floor(room_size_.y / block_size);
		uint block_num_z = (uint)floor(room_size_.z / block_size);

		float jitter_magnitude = (block_size - cell_size_) * 0.8f;
		float velo_magnitude = block_size * 0.02f;
		for (uint i = 0; i < sphere_num_; ++i) {
			uint index = rand() % (block_num_x * block_num_y * block_num_z);
			uint x = index % block_num_x;
			uint z = (index / block_num_x) % block_num_z;
			uint y = (index / block_num_x) / block_num_z;
			h_pos_[i * 3] = origin_.x + block_size * (x + 0.5f) + genJitter(jitter_magnitude);
			h_pos_[i * 3 + 1] = origin_.y + block_size * (y + 0.5f) + genJitter(jitter_magnitude);
			h_pos_[i * 3 + 2] = origin_.z + block_size * (z + 0.5f) + genJitter(jitter_magnitude);
			h_velo_[i * 3] = genJitter(velo_magnitude);
			h_velo_[i * 3 + 1] = genJitter(velo_magnitude);
			h_velo_[i * 3 + 2] = genJitter(velo_magnitude);
			uint proto_id = rand() % PROTOTYPE_NUM;
			h_type_[i] = proto_id;
		}
	}
}

void PhysicsEngine::releaseMemory() {
	delete[] h_pos_;
	delete[] h_velo_;
	delete[] h_type_;

	if (gpu_mode_) {
		freeArray(d_velo_);
		freeArray(d_accel_);
		freeArray(d_pos_);
		freeArray(d_type_);
		freeArray(d_hash_);
		freeArray(d_index_sorted_);
		freeArray(d_cell_start_);
		freeArray(d_cell_end_);
	} 
	else {
		delete[] h_accel_;
		delete[] h_hash_;
		delete[] h_index_sorted_;
		delete[] h_cell_start_;
		delete[] h_cell_end_;
	}
}

float PhysicsEngine::genJitter(float magnitude) {
	return (rand() / (float)RAND_MAX*2.0f - 1.0f) * magnitude;
}

float * PhysicsEngine::outputPos() {
	// if GPU mode, first copy the data from device to host
	if (gpu_mode_) {
		copyDevice2Host(h_pos_, d_pos_, sizeof(float) * 3 * sphere_num_);
	}
	return h_pos_;
}

void PhysicsEngine::update(float elapse) {
	// choose CPU or GPU to do the calculation
	if (gpu_mode_) {
		dSimulateFast(
			d_pos_,
			d_velo_,
			d_accel_,
			d_type_,
			d_hash_,
			d_index_sorted_,
			d_cell_start_,
			d_cell_end_,
			elapse,
			sphere_num_,
			max_hash_value_);
	}
	else {
		if (brutal_mode_) {
			hSimulateBrutal(
				h_pos_,
				h_velo_,
				h_accel_,
				h_type_,
				elapse);
		}
		else {
			hSimulateFast(
				h_pos_,
				h_velo_,
				h_accel_,
				h_type_,
				h_hash_,
				h_index_sorted_,
				h_cell_start_,
				h_cell_end_,
				elapse);
		}
	}
}
