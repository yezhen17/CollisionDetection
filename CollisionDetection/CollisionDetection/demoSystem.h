/*
 * This file defines a demo system for testing the collision detection algorithm
 * This system can both run in render mode and performance test mode
 */

#ifndef DEMOSYSTEM_H
#define DEMOSYSTEM_H

#include <stb_image.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "physicsEngine.h"
#include "shader.h"
#include "camera.h"
#include "sphere.h"

class DemoSystem {
public:
	DemoSystem(bool render_mode = true, 
		bool use_spotlight = false, 
		bool immersive_mode_ = false, 
		float simulation_timestep = 0.1f,
		uint frame_rate = FRAME_RATE, 
		uint sphere_num = SPHERE_NUM, 
		glm::vec3 origin = glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3 room_size = glm::vec3(1.0f, 1.0f, 1.0f));

	~DemoSystem();

	// start demo, including initialization
	// render or not according to render_mode_
	void startDemo();

protected:
	// initialize the glfw window for rendering
	void initWindow();

	//initialize renderer and related data
	void initSpheres();

	// rendering main loop
	void mainLoop();

	// test collision detection algorithm performance without rendering
	void testPerformance(uint test_iters = 1000);

	// whenever the window size changed (by OS or user resize) this callback is called
	void framebufferSizeCallback(int width, int height);

	// whenever the mouse moves, this callback is called
	void mouseCallback(double xpos, double ypos);
	
	// whenever the mouse scroll wheel scrolls, this callback is called
	void scrollCallback(double xoffset, double yoffset);

	// process keyboard input
	void processInput(GLFWwindow *window);

	// update camera position and front for shader
	void updateViewpoint(Shader *shader);

	// render all spheres
	void renderSpheres();

	// render walls and floor
	void renderBackground();

	// load the textures for the wall and floor
	// reference https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/4.2.textures_combined/textures_combined.cpp
	uint loadTexture(char const * path);

	// initialize shader
	Shader *initShader(char const * vs_path, char const * fs_path, uint texture_id, bool has_specular_map);
	
	   
protected:
	// whether open a window and render
	bool render_mode_;

	// gflw window
	GLFWwindow*  window_;

	// window related parameters
	float last_mouse_x_;
	float last_mouse_y_;
	bool first_mouse_;

	// rendering options
	uint frame_rate_;
	float loop_duration_;
	bool use_spotlight_;
	bool immersive_mode_;

	// simulation engine
	PhysicsEngine* engine_;

	// the time elapsed in each step of simulation
	float simulation_timestep_;

	// room origin and size for simulation
	glm::vec3 origin_;
	glm::vec3 room_size_;

	// number of spheres and total number of indices
	uint sphere_num_;
	uint sphere_index_count_;

	// vertex arrays and buffers
	uint sphere_VAO_;
	uint sphere_VBO_;
	uint sphere_EBO_;
	uint background_VAO_;
	uint background_VBO_;
	uint background_EBO_;

	// texture ids
	uint wall_texture_;
	uint floor_texture_;
	
	// camera and shaders
	Camera *camera_;
	Shader *sphere_shader_;
	Shader *background_shader_;
	
    // prestored data for rendering 
	std::vector<glm::vec3> pointlight_positions_;
	std::vector<glm::mat4> model_matrices_;
};

#endif // !DEMOSYSTEM_H

