/*
 * This file defines a demo system for testing the collision detection algorithm
 * This system can both run in render mode and performance test mode
 */

#ifndef DEMOSYSTEM_H
#define DEMOSYSTEM_H

#include "global.h"
#include "physicsEngine.h"

class DemoSystem
{
public:
	DemoSystem(bool render_mode=true, bool use_spotlight=false, bool immersive_mode_=false);
	~DemoSystem();

	// start demo, including initialization
	void startDemo();

protected:
	// initialize necessary parameters and data for the system to work
	void initSystem();

	// initialize the glfw window for rendering
	void initWindow();

	//initialize renderer and related data
	void initRenderer();

	// rendering main loop
	void mainLoop();

	// test collision detection algorithm performance without rendering
	void testPerformance(uint test_iters = 1000);

	// frame buffer callback function
	void framebuffer_size_callback(int width, int height);

	// mouse click callback function
	void mouse_callback(double xpos, double ypos);
	
	// mouse scroll callback function
	void scroll_callback(double xoffset, double yoffset);

	// keyboard callback function
	void processInput(GLFWwindow *window);

	
	void updateShader();
	void updateSpherePosition(float delta_time);

	void loadWallTexture();
	
	   
protected: // data
	// camera
	GLFWwindow*  window_;
	PhysicsEngine* engine_;
	uint sphere_num_;
	uint sphereVAO_;
	uint sphereVBO_;
	uint sphereEBO_;
	GLuint sphere_index_count_;
	Camera *camera_;
	Shader *lighting_shader_;
	float last_mouse_x_;
	float last_mouse_y_;
	bool first_mouse_;

	bool use_spotlight_;

	std::vector<glm::vec3> pointLightPositions;

	// timing
	float deltaTime_;
	float lastFrame_;

	bool render_mode_;
	bool immersive_mode_;
};


#endif // !DEMOSYSTEM_H

