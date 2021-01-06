#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <stb_image.h>

#include "sphere.h"
#include "demoSystem.h"

DemoSystem::DemoSystem()
{
	engine_ = new PhysicsEngine();
	sphere_num_ = SPHERE_NUM;
	/*float timestep = 0.5f;
	float damping = 1.0f;
	float gravity = 0.0003f;

	float collideSpring = 0.5f;;
	float collideDamping = 0.02f;;
	float collideShear = 0.1f;
	engine_->setDrag(damping);
	engine_->setGravity(-gravity);
	engine_->setCollideSpring(collideSpring);
	engine_->setCollideDamping(collideDamping);
	engine_->setCollideShear(collideShear);*/
}

DemoSystem::~DemoSystem()
{

}

void DemoSystem::initSystem()
{
	// camera
	camera_ = Camera(glm::vec3(0.0f, 0.5f, 5.0f));
	lastX_ = WINDOW_WIDTH / 2.0f;
	lastY_ = WINDOW_HEIGHT / 2.0f;
	firstMouse_ = true;

	// timing
	deltaTime_ = 0.0f;
	lastFrame_ = 0.0f;
}

void DemoSystem::initWindow()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Collision", NULL, NULL);
	if (window_ == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window_);
	glfwSetWindowUserPointer(window_, this);
	auto framebuffer_size_callback_func = [](GLFWwindow* window, int width, int height)
	{
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->framebuffer_size_callback(width, height);
	};
	auto mouse_callback_func = [](GLFWwindow* window, double xpos, double ypos)
	{
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->mouse_callback(xpos, ypos);
	};
	auto scroll_callback_func = [](GLFWwindow* window, double xoffset, double yoffset)
	{
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->scroll_callback(xoffset, yoffset);
	};
	glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback_func);
	glfwSetCursorPosCallback(window_, mouse_callback_func);
	glfwSetScrollCallback(window_, scroll_callback_func);

	// tell GLFW to capture our mouse
	// glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);
}

void DemoSystem::initData()
{
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<uint> indices;
	glGenVertexArrays(1, &sphereVAO_);
	glGenBuffers(1, &sphereVBO_);
	glGenBuffers(1, &sphereEBO_);

	for (unsigned int y = 0; y <= Y_SEGMENTS; ++y) {
		for (unsigned int x = 0; x <= X_SEGMENTS; ++x) {
			float xSegment = (float)x / (float)X_SEGMENTS;
			float ySegment = (float)y / (float)Y_SEGMENTS;
			float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
			float yPos = std::cos(ySegment * PI);
			float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
			positions.push_back(glm::vec3(xPos, yPos, zPos));
			normals.push_back(glm::vec3(xPos, yPos, zPos));
		}
	}
	bool odd_row = false;
	for (unsigned int y = 0; y < Y_SEGMENTS; ++y) {
		if (!odd_row) {
			for (unsigned int x = 0; x <= X_SEGMENTS; ++x) {
				indices.push_back(y * (X_SEGMENTS + 1) + x);
				indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
			}
		}
		else {
			for (int x = X_SEGMENTS; x >= 0; --x) {
				indices.push_back((y + 1) * (X_SEGMENTS + 1) + x);
				indices.push_back(y * (X_SEGMENTS + 1) + x);
			}
		}
		odd_row = !odd_row;
	}

	sphere_index_count_ = GLuint(indices.size());
	std::vector<float> sphere_data;
	for (std::size_t i = 0; i < positions.size(); ++i) {
		sphere_data.push_back(positions[i].x);
		sphere_data.push_back(positions[i].y);
		sphere_data.push_back(positions[i].z);
		if (normals.size() > 0) {
			sphere_data.push_back(normals[i].x);
			sphere_data.push_back(normals[i].y);
			sphere_data.push_back(normals[i].z);
		}
	}
	glBindVertexArray(sphereVAO_);
	glBindBuffer(GL_ARRAY_BUFFER, sphereVBO_);
	glBufferData(GL_ARRAY_BUFFER, sphere_data.size() * sizeof(float), &sphere_data[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);	

	// shader configuration
	// --------------------
	
	lighting_shader_.loadProgram("6.sphere.vs", "6.multiple_lights.fs");
	lighting_shader_.use();
	lighting_shader_.setFloat("material.shininess", 32.0f);
	lighting_shader_.setVec3("material.ambient", 0.0f, 0.1f, 0.06f);
	lighting_shader_.setVec3("material.diffuse", 0.0f, 0.50980392f, 0.50980392f);
	lighting_shader_.setVec3("material.specular", 0.50196078f, 0.50196078f, 0.50196078f);
}

void DemoSystem::mainLoop()
{
	glm::vec3 spherePositions[] = {
		glm::vec3(0.0f,  0.0f,  0.0f),
		glm::vec3(2.0f,  5.0f, -15.0f),
		glm::vec3(-1.5f, -2.2f, -2.5f),
		glm::vec3(-3.8f, -2.0f, -12.3f),
		glm::vec3(2.4f, -0.4f, -3.5f),
		glm::vec3(-1.7f,  3.0f, -7.5f),
		glm::vec3(1.3f, -2.0f, -2.5f),
		glm::vec3(1.5f,  2.0f, -2.5f),
		glm::vec3(1.5f,  0.2f, -1.5f),
		glm::vec3(-1.3f,  1.0f, -1.5f)
	};
	pointLightPositions = {
		glm::vec3(0.7f,  0.2f,  2.0f),
		glm::vec3(2.3f, -3.3f, -4.0f),
		glm::vec3(-4.0f,  2.0f, -12.0f),
		glm::vec3(0.0f,  0.0f, -3.0f)
	};
	
	while (!glfwWindowShouldClose(window_))
	{
		// per-frame time logic
		// --------------------
		float currentFrame = glfwGetTime();
		deltaTime_ = currentFrame - lastFrame_;
		lastFrame_ = currentFrame;

		printf("%.6f", deltaTime_);
		//Sleep(200);
		
		// input
		// -----
		processInput(window_);

		// render
		// ------
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		updateShader();

		// view/projection transformations
		glm::mat4 projection = glm::perspective(glm::radians(camera_.Zoom), (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera_.GetViewMatrix();
		lighting_shader_.setMat4("projection", projection);
		lighting_shader_.setMat4("view", view);

		// world transformation
		glm::mat4 model = glm::mat4(1.0f);
		lighting_shader_.setMat4("model", model);

		for (int i = 0; i < 1; i++) {
			updateSpherePosition(deltaTime_);
		}
		
		glfwSwapBuffers(window_);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &sphereVAO_);
	glDeleteBuffers(1, &sphereVBO_);
	glDeleteBuffers(1, &sphereEBO_);

	glfwTerminate();
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void DemoSystem::processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera_.ProcessKeyboard(FORWARD, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera_.ProcessKeyboard(BACKWARD, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera_.ProcessKeyboard(LEFT, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera_.ProcessKeyboard(RIGHT, deltaTime_);
}

void DemoSystem::updateShader()
{
	// be sure to activate shader when setting uniforms/drawing objects
	lighting_shader_.use();
	lighting_shader_.setVec3("viewPos", camera_.Position);

	/*
	   Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
	   the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
	   by defining light types as classes and set their values in there, or by using a more efficient uniform approach
	   by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
	*/
	// directional light
	lighting_shader_.setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
	lighting_shader_.setVec3("dirLight.ambient", 0.05f, 0.05f, 0.05f);
	lighting_shader_.setVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
	lighting_shader_.setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);
	// point light 1
	lighting_shader_.setVec3("pointLights[0].position", pointLightPositions[0]);
	lighting_shader_.setVec3("pointLights[0].ambient", 0.05f, 0.05f, 0.05f);
	lighting_shader_.setVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
	lighting_shader_.setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setFloat("pointLights[0].constant", 1.0f);
	lighting_shader_.setFloat("pointLights[0].linear", 0.09);
	lighting_shader_.setFloat("pointLights[0].quadratic", 0.032);
	// point light 2
	lighting_shader_.setVec3("pointLights[1].position", pointLightPositions[1]);
	lighting_shader_.setVec3("pointLights[1].ambient", 0.05f, 0.05f, 0.05f);
	lighting_shader_.setVec3("pointLights[1].diffuse", 0.8f, 0.8f, 0.8f);
	lighting_shader_.setVec3("pointLights[1].specular", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setFloat("pointLights[1].constant", 1.0f);
	lighting_shader_.setFloat("pointLights[1].linear", 0.09);
	lighting_shader_.setFloat("pointLights[1].quadratic", 0.032);
	// point light 3
	lighting_shader_.setVec3("pointLights[2].position", pointLightPositions[2]);
	lighting_shader_.setVec3("pointLights[2].ambient", 0.05f, 0.05f, 0.05f);
	lighting_shader_.setVec3("pointLights[2].diffuse", 0.8f, 0.8f, 0.8f);
	lighting_shader_.setVec3("pointLights[2].specular", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setFloat("pointLights[2].constant", 1.0f);
	lighting_shader_.setFloat("pointLights[2].linear", 0.09);
	lighting_shader_.setFloat("pointLights[2].quadratic", 0.032);
	// point light 4
	lighting_shader_.setVec3("pointLights[3].position", pointLightPositions[3]);
	lighting_shader_.setVec3("pointLights[3].ambient", 0.05f, 0.05f, 0.05f);
	lighting_shader_.setVec3("pointLights[3].diffuse", 0.8f, 0.8f, 0.8f);
	lighting_shader_.setVec3("pointLights[3].specular", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setFloat("pointLights[3].constant", 1.0f);
	lighting_shader_.setFloat("pointLights[3].linear", 0.09);
	lighting_shader_.setFloat("pointLights[3].quadratic", 0.032);
	// spotLight
	lighting_shader_.setVec3("spotLight.position", camera_.Position);
	lighting_shader_.setVec3("spotLight.direction", camera_.Front);
	lighting_shader_.setVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
	lighting_shader_.setVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
	lighting_shader_.setFloat("spotLight.constant", 1.0f);
	lighting_shader_.setFloat("spotLight.linear", 0.09);
	lighting_shader_.setFloat("spotLight.quadratic", 0.032);
	lighting_shader_.setFloat("spotLight.cutOff", glm::cos(glm::radians(12.5f)));
	lighting_shader_.setFloat("spotLight.outerCutOff", glm::cos(glm::radians(15.0f)));
}

void DemoSystem::updateSpherePosition(float delta_time)
{
	float timestep = 0.1f;//0.5f
	//float damping = 0.999f;
	//float gravity = 0.05f;//0.001f;

	//float collideSpring = 2.5f;//0.5f;
	//float collideDamping = 0.02f;
	//float collideShear = 0.1f;
	//float collideE = 0.2f;
	//engine_->setDrag(damping);
	//engine_->setGravity(-gravity);
	//engine_->setCollideSpring(collideSpring);
	//engine_->setCollideDamping(collideDamping);
	//engine_->setCollideShear(collideShear);
	//engine_->setCollideE(collideE);
	engine_->update(timestep);
	
	float* updated_pos = engine_->outputPos();
	float *sphere_radius = engine_->getSphereRadius();
	// draw spheres
	glBindVertexArray(sphereVAO_);
	// glDrawElements(GL_TRIANGLE_STRIP, sphereIndexCount, GL_UNSIGNED_INT, 0);
	for (uint i = 0; i < sphere_num_; ++i)
	{
		// calculate the model matrix for each object and pass it to shader before drawing
		glm::mat4 model = glm::mat4(1.0f);
		uint i_x3 = i * 3;
		model = glm::translate(model, glm::vec3(updated_pos[i_x3], updated_pos[i_x3 +1], updated_pos[i_x3 +2]));
		float angle = 20.0f * i;
		model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
		lighting_shader_.setMat4("model", model);
		lighting_shader_.setFloat("radius", sphere_radius[i]);
		glDrawElements(GL_TRIANGLE_STRIP, sphere_index_count_, GL_UNSIGNED_INT, 0);
	}
	//system("pause");
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void DemoSystem::framebuffer_size_callback(int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void DemoSystem::mouse_callback(double xpos, double ypos)
{
	if (firstMouse_)
	{
		lastX_ = xpos;
		lastY_ = ypos;
		firstMouse_ = false;
	}

	float xoffset = xpos - lastX_;
	float yoffset = lastY_ - ypos; // reversed since y-coordinates go from bottom to top

	lastX_ = xpos;
	lastY_ = ypos;

	camera_.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void DemoSystem::scroll_callback(double xoffset, double yoffset)
{
	camera_.ProcessMouseScroll(yoffset);
}