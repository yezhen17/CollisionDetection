/*
 * This header file defines the settings as constants
 */

#ifndef GLOBAL_H
#define GLOBAL_H

typedef unsigned int uint;
typedef unsigned char uchar;

// fixed
// ---------------------------------------------------------------------------------------

const uint HASH_BLOCK = 64; // fix this since 64 is suitable for this task
const float PI = 3.14159265359f;
const uint PROTOTYPE_NUM = 7;  // maximum 7 prototypes

// ---------------------------------------------------------------------------------------

// can be modified
// ---------------------------------------------------------------------------------------

const float MOUSE_SPEED = 2.5f;
const float ZOOM_DEFAULT = 20.0f;

const uint SPHERE_NUM = 233;
const uint FRAME_RATE = 25;

const bool GPU_MODE = true;

const uint WINDOW_WIDTH = 1600;
const uint WINDOW_HEIGHT = 900;

const uint HORIZONTAL_FRAGMENT_NUM = 16;
const uint VERTICAL_FRAGMENT_NUM = 16;

// ---------------------------------------------------------------------------------------

#endif



