#ifndef SPHERE_H
#define SPHERE_H

#include "global.h"

typedef struct Sphere
{
	float mass;
	float radius;
	float rest;
	float r;
	float g;
	float b;

	Sphere(float mass, float radius, float rest, float r, float g, float b) :
		mass(mass),
		radius(radius),
		rest(rest),
		r(r),
		g(g),
		b(b)
	{

	}
} Sphere;

const uint PROTOTYPE_NUM = 7;
const Sphere PROTOTYPES[7] =
{
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(2.0f, 1.0f / 24.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 24.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 24.0f, 0.5f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0.7f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 1.0f, 0.0f, 0.0f, 0.0f)
};

#endif