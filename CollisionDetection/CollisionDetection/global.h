/*
 * This header file defines the settings as constants
 */

#ifndef GLOBAL_H
#define GLOBAL_H

typedef unsigned int uint;

// cannot be changed by user
// ---------------------------------------------------------------------------------------

const uint HASH_BLOCK = 64; // fix this since 64 is suitable for this task
const float PI = 3.14159265359f;
const uint PROTOTYPE_NUM = 7;  // maximum 7 prototypes

const float MOUSE_SPEED = 5.0f;
const float ZOOM_DEFAULT = 20.0f;

// determines how "round" the sphere is: the larger, the rounder
const uint HORIZONTAL_FRAGMENT_NUM = 16;
const uint VERTICAL_FRAGMENT_NUM = 16;

const uint WINDOW_WIDTH = 800;
const uint WINDOW_HEIGHT = 800;

const uint FRAME_RATE = 50;

const bool VERBOSE = true;

const bool BRUTAL_MODE = false;

// ---------------------------------------------------------------------------------------

// can be modified by user
// ---------------------------------------------------------------------------------------

const uint SPHERE_NUM_DEFAULT = 10;

const bool RENDER_MODE_DEFAULT = true;
const bool GPU_MODE_DEFAULT = true;

// ---------------------------------------------------------------------------------------

#endif



