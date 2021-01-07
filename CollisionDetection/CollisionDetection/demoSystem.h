#ifndef DEMOSYSTEM_H
#define DEMOSYSTEM_H

#include "global.h"
#include "physicsEngine.h"

class DemoSystem
{
public:
	DemoSystem();
	~DemoSystem();
	void initWindow();
	void initSystem();
	void initData();
	void mainLoop();
	void framebuffer_size_callback(int width, int height);
	void mouse_callback(double xpos, double ypos);
	void scroll_callback(double xoffset, double yoffset);
	void processInput(GLFWwindow *window);
	void testPerformance(uint test_iters = 1000);
	   
protected: // data
	// camera
	GLFWwindow*  window_;
	PhysicsEngine* engine_;
	uint sphere_num_;
	uint sphereVAO_;
	uint sphereVBO_;
	uint sphereEBO_;
	GLuint sphere_index_count_;
	Camera camera_;
	Shader lighting_shader_;
	float lastX_;
	float lastY_;
	bool firstMouse_;

	std::vector<glm::vec3> pointLightPositions;

	// timing
	float deltaTime_;
	float lastFrame_;

private:
	void updateShader();
	void updateSpherePosition(float delta_time);


};


#endif // !DEMOSYSTEM_H

