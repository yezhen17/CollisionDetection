#ifndef SPHERE_H
#define SPHERE_H

#include "global.h"

typedef struct Sphere {
	float mass;
	float radius;
	uint material_type;
	float r;
	float g;
	float b;
	float stiffness;

	Sphere(float mass, float radius, uint material_type, float r, float g, float b, float stiffness=1.0f) :
		mass(mass),
		radius(radius),
		material_type(material_type),
		r(r),
		g(g),
		b(b),
		stiffness(stiffness) {

	}
} Sphere;

const Sphere PROTOTYPES[7] = {
	Sphere(1.0f, 1.0f / 32.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(2.0f, 1.0f / 24.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 24.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 24.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0, 0.0f, 0.0f, 0.0f),
	Sphere(1.0f, 1.0f / 32.0f, 0, 0.0f, 0.0f, 0.0f)
};

#endif