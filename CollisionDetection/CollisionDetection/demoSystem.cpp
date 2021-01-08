/*
 * Implementation of the demo system
 */

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"
#include "camera.h"
#include <stb_image.h>

#include <windows.h>
#include <windef.h>

#include "sphere.h"
#include "demoSystem.h"

DemoSystem::DemoSystem(bool render_mode, bool use_spotlight, bool immersive_mode, uint frame_rate, uint sphere_num):
	render_mode_(render_mode),
	use_spotlight_(use_spotlight),
	immersive_mode_(immersive_mode),
	frame_rate_(frame_rate),
	loop_duration_(1.0f / frame_rate),
	sphere_num_(sphere_num) {
	engine_ = new PhysicsEngine();
}

DemoSystem::~DemoSystem() {

}

void DemoSystem::startDemo() {
	initSystem();
	if (render_mode_)
	{
		initWindow();
		initRenderer();
		mainLoop();
	}
	else
	{
		testPerformance();
	}
}

void DemoSystem::initSystem() {
	// camera
	camera_ = new Camera(5, -20, 225);
	last_mouse_x_ = WINDOW_WIDTH / 2.0f;
	last_mouse_y_ = WINDOW_HEIGHT / 2.0f;
	first_mouse_ = true;

	// timing
	deltaTime_ = 0.0f;
	lastFrame_ = 0.0f;
}

void DemoSystem::initWindow() {
	// glfw initialization and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	window_ = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Demo", NULL, NULL);
	if (window_ == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(window_);

	// set callback functions
	// need this conversion because the original callback functions are class members
	// reference: https://stackoverflow.com/questions/7676971/pointing-to-a-function-that-is-a-class-member-glfw-setkeycallback
	glfwSetWindowUserPointer(window_, this);
	auto framebuffer_size_callback_func = [](GLFWwindow* window, int width, int height) {
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->framebuffer_size_callback(width, height);
	};
	auto mouse_callback_func = [](GLFWwindow* window, double xpos, double ypos) {
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->mouse_callback(xpos, ypos);
	};
	auto scroll_callback_func = [](GLFWwindow* window, double xoffset, double yoffset) {
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->scroll_callback(xoffset, yoffset);
	};

	glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback_func);
	glfwSetCursorPosCallback(window_, mouse_callback_func);
	glfwSetScrollCallback(window_, scroll_callback_func);

	if (immersive_mode_) {
		// tell glfw to capture our mouse
		glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
}

void DemoSystem::initRenderer() {
	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);

	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<uint> indices;
	glGenVertexArrays(1, &sphere_VAO_);
	glGenBuffers(1, &sphere_VBO_);
	glGenBuffers(1, &sphere_EBO_);

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

	sphere_index_count_ = indices.size();
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
	glBindVertexArray(sphere_VAO_);
	glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO_);
	glBufferData(GL_ARRAY_BUFFER, sphere_data.size() * sizeof(float), &sphere_data[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// set up vertex data (and buffer(s)) and configure vertex attributes
	// ------------------------------------------------------------------
	float wall_vertices[] = {
		// positions          // colors           // texture coords
		 0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
		 0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
		-0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
		-0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
	};
	uint wall_indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};

	glGenVertexArrays(1, &wall_VAO_);
	glGenBuffers(1, &wall_VBO_);
	glGenBuffers(1, &wall_EBO_);

	glBindVertexArray(wall_VAO_);

	glBindBuffer(GL_ARRAY_BUFFER, wall_VBO_);
	glBufferData(GL_ARRAY_BUFFER, sizeof(wall_vertices), wall_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wall_EBO_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(wall_indices), wall_indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	/*
   Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
   the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
   by defining light types as classes and set their values in there, or by using a more efficient uniform approach
   by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
*/

	pointlight_positions_ = {
		glm::vec3(0.0f,  1.5f,  0.0f),
	};

	wall_positions_ = {
		glm::vec3(-1.0f,  0.0f,  -1.0f),
		glm::vec3(-1.0f,  2.0f,  -1.0f),
		glm::vec3(-1.0f,  1.0f,  -1.0f),
	};

	wall_texture_ = loadTexture("../resources/textures/brickwall.jpg");
	floor_texture_ = loadTexture("../resources/textures/wood.png");
	wall_shader_ = initShader("../shaderPrograms/background.vs", "../shaderPrograms/phong.fs", wall_texture_);
	sphere_shader_ = initShader("../shaderPrograms/sphere.vs", "../shaderPrograms/phong.fs", 0);
}

Shader *DemoSystem::initShader(char const * vs_path, char const * fs_path, uint texture_id)
{
	Shader *shader = new Shader();
	// shader configuration
	shader->loadProgram(vs_path, fs_path);
	shader->use();
	shader->setFloat("material.shininess", 32.0f);
	shader->setVec3("material.ambient", 0.0f, 0.1f, 0.06f);
	shader->setVec3("material.diffuse", 0.0f, 0.50980392f, 0.50980392f);
	shader->setVec3("material.specular", 0.50196078f, 0.50196078f, 0.50196078f);

	// point light 1
	shader->setVec3("pointLights[0].position", pointlight_positions_[0]);
	shader->setVec3("pointLights[0].ambient", 0.1f, 0.1f, 0.1f);
	shader->setVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
	shader->setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
	shader->setFloat("pointLights[0].constant", 1.0f);
	shader->setFloat("pointLights[0].linear", 0.09);
	shader->setFloat("pointLights[0].quadratic", 0.032);

	// directional light
	//shader->setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
	shader->setVec3("dirLight.direction", 0.0f, -1.0f, 0.0f);
	shader->setVec3("dirLight.ambient", 0.1f, 0.1f, 0.1f);
	shader->setVec3("dirLight.diffuse", 0.7f, 0.7f, 0.7f);
	shader->setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);

	bool has_texture = texture_id != 0;
	std::cout << std::endl << has_texture;
	shader->setBool("HasTexture", has_texture);

	if (use_spotlight_) {
		// spotlight from camera
		shader->setVec3("spotLight.ambient", 0.0f, 0.0f, 0.0f);
		shader->setVec3("spotLight.diffuse", 1.0f, 1.0f, 1.0f);
		shader->setVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
		shader->setFloat("spotLight.constant", 1.0f);
		shader->setFloat("spotLight.linear", 0.09);
		shader->setFloat("spotLight.quadratic", 0.032);
		shader->setFloat("spotLight.cutOff", glm::cos(glm::radians(12.5f)));
		shader->setFloat("spotLight.outerCutOff", glm::cos(glm::radians(15.0f)));
	}

	if (has_texture) {
		shader->setInt("material.diffuseMap", texture_id);
	}

	return shader;
}

void DemoSystem::mainLoop() {
	while (!glfwWindowShouldClose(window_)) {
		// per-frame time logic
		// --------------------
		float time_start = glfwGetTime();
		/*deltaTime_ = currentFrame - lastFrame_;
		lastFrame_ = currentFrame;*/

		//printf("%.6f", deltaTime_);
		//Sleep(200);

		processInput(window_);

		// render
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		renderBackground();
		renderSpheres();

		glfwSwapBuffers(window_);
		glfwPollEvents();
		float time_end = glfwGetTime();
		float time_elapse = time_end - time_start;
		Sleep(time_elapse * 1000);
	}

	glDeleteVertexArrays(1, &sphere_VAO_);
	glDeleteVertexArrays(1, &wall_VAO_);
	glDeleteBuffers(1, &sphere_VBO_);
	glDeleteBuffers(1, &sphere_EBO_);
	glDeleteBuffers(1, &wall_VBO_);
	glDeleteBuffers(1, &wall_EBO_);

	glfwTerminate();
}



void DemoSystem::renderBackground() {
	// bind textures on corresponding texture units
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, wall_texture_);
	wall_shader_->use();
	updateViewpoint(wall_shader_);
	glBindVertexArray(wall_VAO_);
	
	for (unsigned int i = 0; i < 2; i++)
	{
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, wall_positions_[i]);
		model = glm::scale(model, glm::vec3(0.5f)); // Make it a smaller cube
		wall_shader_->setMat4("model", model);
		// glDrawArrays(GL_TRIANGLES, 0, 36);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
		/*wall_shader_->setInt("material.diffuseMap", floor_texture_);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, floor_texture_);
		
		glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (void *)(GL_UNSIGNED_INT*3));*/
	}
	glBindTexture(GL_TEXTURE_2D, floor_texture_);
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, wall_positions_[2]);
	model = glm::scale(model, glm::vec3(0.5f)); // Make it a smaller cube
	wall_shader_->setMat4("model", model);
	// glDrawArrays(GL_TRIANGLES, 0, 36);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);

}

void DemoSystem::renderSpheres() {
	sphere_shader_->use();
	updateViewpoint(sphere_shader_);

	float currentFrame1 = glfwGetTime();

	float timestep = 0.1f;//0.5f
	engine_->update(timestep);

	float* updated_pos = engine_->outputPos();

	float currentFrame2 = glfwGetTime();
	//printf("%.6f", currentFrame2 - currentFrame1);

	uint *type = engine_->getSphereType();
	// draw spheres
	glBindVertexArray(sphere_VAO_);
	// glDrawElements(GL_TRIANGLE_STRIP, sphereIndexCount, GL_UNSIGNED_INT, 0);
	for (uint i = 0; i < sphere_num_; ++i)
	{
		// calculate the model matrix for each object and pass it to shader before drawing
		glm::mat4 model = glm::mat4(1.0f);
		uint i_x3 = i * 3;
		model = glm::translate(model, glm::vec3(updated_pos[i_x3], updated_pos[i_x3 + 1], updated_pos[i_x3 + 2]));
		float angle = 20.0f * i;
		model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
		sphere_shader_->setMat4("model", model);
		sphere_shader_->setFloat("radius", PROTOTYPES[type[i]].radius);
		glDrawElements(GL_TRIANGLE_STRIP, sphere_index_count_, GL_UNSIGNED_INT, 0);
	}
}

void DemoSystem::testPerformance(uint test_iters)
{
	LARGE_INTEGER frequency, startCount, stopCount;
	BOOL ret;
	//返回性能计数器每秒滴答的个数
	ret = QueryPerformanceFrequency(&frequency);
	if (ret) {
		ret = QueryPerformanceCounter(&startCount);
	}
	for (uint i = 0; i < test_iters; ++i)
	{
		engine_->update(0.1f);
	}
	if (ret) {
		QueryPerformanceCounter(&stopCount);
	}
	if (ret) {
		LONGLONG elapsed = (stopCount.QuadPart - startCount.QuadPart) * 1000000 / frequency.QuadPart;
		printf("QueryPerformanceFrequency & QueryPerformanceCounter = %ld us", elapsed);
	}

}

void DemoSystem::updateViewpoint(Shader *shader) {
	// camera position update
	shader->setVec3("viewPos", camera_->getCameraPos());

	if (use_spotlight_) {
		// spotLight position and direction update
		shader->setVec3("spotLight.position", camera_->getCameraPos());
		shader->setVec3("spotLight.direction", camera_->getCameraFront());
	}

	// view/projection transformations
	glm::mat4 projection = glm::perspective(glm::radians(camera_->getZoom()), (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 100.0f);
	glm::mat4 view = camera_->GetViewMatrix();
	shader->setMat4("projection", projection);
	shader->setMat4("view", view);
}

uint DemoSystem::loadTexture(char const * path) {
	unsigned int textureID;
	glGenTextures(1, &textureID);

	int width, height, nrComponents;
	unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
	if (data)
	{
		GLenum format;
		if (nrComponents == 1)
			format = GL_RED;
		else if (nrComponents == 3)
			format = GL_RGB;
		else if (nrComponents == 4)
			format = GL_RGBA;

		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(data);
	}
	else
	{
		std::cout << "Texture failed to load at path: " << path << std::endl;
		stbi_image_free(data);
	}

	return textureID;
}

// process all input: query glfw whether relevant keys are pressed/released this frame and react accordingly
void DemoSystem::processInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera_->ProcessKeyboard(FORWARD, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera_->ProcessKeyboard(BACKWARD, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera_->ProcessKeyboard(LEFT, deltaTime_);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera_->ProcessKeyboard(RIGHT, deltaTime_);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void DemoSystem::framebuffer_size_callback(int width, int height) {
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
void DemoSystem::mouse_callback(double pos_x, double pos_y) {
	if (first_mouse_) {
		last_mouse_x_ = pos_x;
		last_mouse_y_ = pos_y;
		first_mouse_ = false;
	}

	float xoffset = pos_x - last_mouse_x_;
	// reversed since y-coordinates go from bottom to top
	float yoffset = last_mouse_y_ - pos_y;
	last_mouse_x_ = pos_x;
	last_mouse_y_ = pos_y;

	camera_->ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
void DemoSystem::scroll_callback(double xoffset, double yoffset) {
	camera_->ProcessMouseScroll(yoffset);
}