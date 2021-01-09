/*
 * This file defines a struct called Sphere that serve as the prototypes
 * for the value of the materials: http://devernay.free.fr/cours/opengl/materials.html
 */

#ifndef SPHERE_H
#define SPHERE_H

#include <glm/glm.hpp>

typedef unsigned int uint;

typedef struct Sphere {
	float mass;
	float radius;
	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;
	float shininess;
	float stiffness;

	Sphere(float mass, float radius, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular, float shininess, float stiffness=1.0f) :
		mass(mass),
		radius(radius),
		ambient(ambient),
		diffuse(diffuse),
		specular(specular),
		shininess(shininess),
		stiffness(stiffness) {

	}
} Sphere;

// TODO
const Sphere PROTOTYPES[7] = {
	// ruby
	Sphere(1.0f, 1.0f / 32.0f, 
	             glm::vec3(0.1745f, 0.01175f, 0.01175f), 
				 glm::vec3(0.61424f, 0.04136f, 0.04136f),
				 glm::vec3(0.727811f, 0.626959f, 0.626959f),
				 76.8f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.1745f, 0.01175f, 0.01175f),
				 glm::vec3(0.61424f, 0.04136f, 0.04136f),
				 glm::vec3(0.727811f, 0.626959f, 0.626959f),
				 76.8f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.1745f, 0.01175f, 0.01175f),
				 glm::vec3(0.61424f, 0.04136f, 0.04136f),
				 glm::vec3(0.727811f, 0.626959f, 0.626959f),
				 76.8f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.1745f, 0.01175f, 0.01175f),
				 glm::vec3(0.61424f, 0.04136f, 0.04136f),
				 glm::vec3(0.727811f, 0.626959f, 0.626959f),
				 76.8f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.1745f, 0.01175f, 0.01175f),
				 glm::vec3(0.61424f, 0.04136f, 0.04136f),
				 glm::vec3(0.727811f, 0.626959f, 0.626959f),
				 76.8f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.0f, 0.1f, 0.06f),
				 glm::vec3(0.0f, 0.50980392f, 0.50980392f),
				 glm::vec3(0.50196078f, 0.50196078f, 0.50196078f),
				 32.0f),
	Sphere(1.0f, 1.0f / 32.0f,
				 glm::vec3(0.0f, 0.1f, 0.06f),
				 glm::vec3(0.0f, 0.50980392f, 0.50980392f),
				 glm::vec3(0.50196078f, 0.50196078f, 0.50196078f),
				 32.0f),
};

struct SimulationSphereProto {
	float radii[7];
	float masses[7];
};

#endif