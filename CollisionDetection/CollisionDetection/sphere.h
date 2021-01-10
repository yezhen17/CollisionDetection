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
	float shininess;
	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;
	
	Sphere(float mass, float radius, float shininess, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular) :
		mass(mass),
		radius(radius),
		ambient(ambient),
		diffuse(diffuse),
		specular(specular),
		shininess(shininess) {

	}
} Sphere;

const Sphere PROTOTYPES[4] = {
	// ruby
	Sphere(1.0f, 1.0f / 32.0f, 76.8f,
	             glm::vec3(0.1745f, 0.01175f, 0.01175f), 
				 glm::vec3(0.61424f, 0.04136f, 0.04136f), 
	             glm::vec3(0.727811f, 0.626959f, 0.626959f)),
	// emerald
	Sphere(1.0f, 1.0f / 32.0f, 76.8f,
				 glm::vec3(0.0215f, 0.1745f, 0.0215f),
				 glm::vec3(0.07568f, 0.61424f, 0.07568f),
				 glm::vec3(0.633f, 0.727811f, 0.633f)),
	// silver
	Sphere(1.5f, 1.0f / 48.0f, 51.2f,
				 glm::vec3(0.19225f, 0.19225f, 0.19225f),
				 glm::vec3(0.50754f, 0.50754f, 0.50754f),
				 glm::vec3(0.508273f, 0.508273f, 0.508273f)),
	// gold
	Sphere(2.0f, 1.0f / 64.0f, 51.2f,
				 glm::vec3(0.24725f, 0.1995f, 0.0745f),
				 glm::vec3(0.75164f, 0.60648f, 0.22648f),
				 glm::vec3(0.628281f, 0.555802f, 0.366065f))
};

// restitution coefficient when two prototypes collide
const float RESTITUTION[4][4] = {
	1.0f, 1.0f, 0.8f, 0.9f,
	1.0f, 1.0f, 0.8f, 0.9f,
	0.8f, 0.8f, 0.5f, 0.6f,
	0.9f, 0.9f, 0.6f, 0.7f,
	/*1.0f, 0.9f, 0.8f, 0.7f,
	0.9f, 0.8f, 0.7f, 0.6f,
	0.8f, 0.7f, 0.6f, 0.5f,
	0.7f, 0.6f, 0.5f, 0.4f,*/
};

#endif