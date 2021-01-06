#ifndef SPHERE_H
#define SPHERE_H

#include "global.h"

typedef struct Sphere
{
	float mass_;
	float radius_;
	float rest_;
	float r_;
	float g_;
	float b_;

	Sphere(float mass, float radius, float rest, float r, float g, float b) :
		mass_(mass),
		radius_(radius),
		rest_(rest),
		r_(r),
		g_(g),
		b_(b)
	{

	}
} Sphere;

const uint PROTOTYPE_NUM = 7;
const Sphere PROTOTYPES[7] =
{
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0.5f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0.7f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f)
};

#endif