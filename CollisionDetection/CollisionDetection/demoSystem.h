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
	DemoSystem(bool render_mode=true, bool use_spotlight=false, bool immersive_mode_=false,
		uint frame_rate=FRAME_RATE, uint sphere_num=SPHERE_NUM);
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
	void framebufferSizeCallback(int width, int height);

	// mouse click callback function
	void mouseCallback(double xpos, double ypos);
	
	// mouse scroll callback function
	void scrollCallback(double xoffset, double yoffset);

	// keyboard callback function
	void processInput(GLFWwindow *window);

	
	void updateViewpoint(Shader *shader);
	void renderSpheres();
	void renderBackground();

	// load the textures for the wall and floor
	// reference https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/4.2.textures_combined/textures_combined.cpp
	uint loadTexture(char const * path);

	// initialize shader
	Shader *initShader(char const * vs_path, char const * fs_path, uint texture_id);
	
	   
protected: // data
	// camera
	GLFWwindow*  window_;
	PhysicsEngine* engine_;
	uint sphere_num_;
	uint sphere_VAO_;
	uint sphere_VBO_;
	uint sphere_EBO_;

	uint wall_VAO_;
	uint wall_VBO_;
	uint wall_EBO_;
	uint wall_texture_;
	uint floor_texture_;
	uint sphere_index_count_;
	Camera *camera_;
	Shader *sphere_shader_;
	Shader *wall_shader_;
	float last_mouse_x_;
	float last_mouse_y_;
	bool first_mouse_;

	uint frame_rate_;
	float loop_duration_;

	bool use_spotlight_;

	std::vector<glm::vec3> pointlight_positions_;
	std::vector<glm::vec3> wall_positions_;

	// timing
	float deltaTime_;
	float lastFrame_;

	bool render_mode_;
	bool immersive_mode_;
};


#endif // !DEMOSYSTEM_H

