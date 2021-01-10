/*
 * This header file defines the settings as constants
 */

#ifndef GLOBAL_H
#define GLOBAL_H

typedef unsigned int uint;

// cannot be changed by user
// ---------------------------------------------------------------------------------------

const float PI = 3.141592653f;
const float E = 2.718281828f;

const uint HASH_BLOCK = 64; // fix this since 64 is suitable for this task
const uint PROTOTYPE_NUM = 4;  // maximum 4 prototypes

const float MOUSE_SPEED = 5.0f;
const float ZOOM_DEFAULT = 20.0f;

// determines how "round" the sphere is: the larger, the rounder
const uint HORIZONTAL_FRAGMENT_NUM = 12;
const uint VERTICAL_FRAGMENT_NUM = 12;

const uint WINDOW_WIDTH = 800;
const uint WINDOW_HEIGHT = 800;

const uint FRAME_RATE = 50; // how many rendering frames each second
const uint SIMULATION_STEP = 3; // how many steps to execute each rendering frame

const bool VERBOSE = false; // whether remind user the frame rate is too high

const bool BRUTAL_MODE = false; // if use CPU mode, whether use brutal-force algorithm
const bool IMMERSIVE_MODE = false; // whether hide the cursor
const bool USE_SPOTLIGHT = false; // whether use spotlight (disable for less computation)

// environment related, PLEASE do not modify these
const float DRAG = 0.9999f;
const float GRAVITY = 0.2f;
const float STIFFNESS = 1000.0f;
const float DAMPING = 8.0f;
const float FRICTION = 0.05f;
const float SIMULATION_TIMESTEP = 0.02f; // time elapsed in one "update"

// ---------------------------------------------------------------------------------------

// can be modified by user
// ---------------------------------------------------------------------------------------

const uint SPHERE_NUM_DEFAULT = 1000;

const bool RENDER_MODE_DEFAULT = true;
const bool GPU_MODE_DEFAULT = true;

// ---------------------------------------------------------------------------------------

#endif



