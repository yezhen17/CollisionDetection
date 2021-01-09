/*
  * implementation of the z-order curve (or Morton encoding)
  * reference: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
  * reference: https://john.cs.olemiss.edu/~rhodes/papers/Nocentino10.pdf
  */

#ifndef MORTONENCODE_CUH
#define MORTONENCODE_CUH

#include <stdio.h>
#include <math.h>
#include <helper_math.h>
#include <math_constants.h>
#include <device_launch_parameters.h>


// "insert" a 0 bit after each of the 16 low bits of x
__forceinline__ __device__ uint dPart1By1(uint x)
{
	x &= 0x0000003f; // x = ---- ---- ---- ---- ---- ---- --54 3210
	x = (x ^ (x << 4)) & 0x00000f0f; // x = ---- ---- ---- ---- ---- --54 ---- 3210
	x = (x ^ (x << 2)) & 0x00003333; // x = ---- ---- ---- ---- ---- --54 --32 --10
	x = (x ^ (x << 1)) & 0x00005555; // x = ---- ---- ---- ---- ---- -5-4 -3-2 -1-0
	return x;
}

// "insert" two 0 bits after each of the 10 low bits of x
__forceinline__ __device__ uint dPart1By2(uint x)
{
	x &= 0x0000003f; // x = ---- ---- ---- ---- ---- ---- --54 3210
	x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- ---- ---- ---- --54 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- ---- ---- ---- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249; // x = ---- ---- ---- ---- 5--4 --3- -2-- 1--0
	return x;
}

__forceinline__ __device__ uint dMortonEncode2D(int2 pos)
{
	return (dPart1By1(pos.y) << 1) + dPart1By1(pos.x);
}

// only applicable for HASH_BLOCK = 64
__forceinline__ __device__ uint dMortonEncode3D(int3 pos)
{
	return (dPart1By2(pos.z) << 2) + (dPart1By2(pos.y) << 1) + dPart1By2(pos.x);
}

// CPU versions
inline uint hPart1By1(uint x)
{
	x &= 0x0000003f; // x = ---- ---- ---- ---- ---- ---- --54 3210
	x = (x ^ (x << 4)) & 0x00000f0f; // x = ---- ---- ---- ---- ---- --54 ---- 3210
	x = (x ^ (x << 2)) & 0x00003333; // x = ---- ---- ---- ---- ---- --54 --32 --10
	x = (x ^ (x << 1)) & 0x00005555; // x = ---- ---- ---- ---- ---- -5-4 -3-2 -1-0
	return x;
}

inline uint hPart1By2(uint x)
{
	x &= 0x0000003f; // x = ---- ---- ---- ---- ---- ---- --54 3210
	x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- ---- ---- ---- --54 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- ---- ---- ---- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249; // x = ---- ---- ---- ---- 5--4 --3- -2-- 1--0
	return x;
}

inline uint hMortonEncode2D(int2 pos)
{
	return (hPart1By1(pos.y) << 1) + hPart1By1(pos.x);
}

// only applicable for HASH_BLOCK = 64
inline uint hMortonEncode3D(int3 pos)
{
	return (hPart1By2(pos.z) << 2) + (hPart1By2(pos.y) << 1) + hPart1By2(pos.x);
}

#endif // !MORTONENCODE_CUH