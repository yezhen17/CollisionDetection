/*
 * Implementation of the demo system
 */

#include <iostream>
#include <windows.h>
#include <windef.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "demoSystem.h"

DemoSystem::DemoSystem(bool render_mode, bool use_spotlight, bool immersive_mode, float simulation_timestep,
	uint frame_rate, uint sphere_num, glm::vec3 origin, glm::vec3 room_size):
	render_mode_(render_mode),
	use_spotlight_(use_spotlight),
	immersive_mode_(immersive_mode),
	simulation_timestep_(simulation_timestep),
	frame_rate_(frame_rate),
	loop_duration_(1.0f / frame_rate),
	sphere_num_(sphere_num),
	origin_(origin),
	room_size_(room_size) {
	engine_ = new PhysicsEngine(sphere_num, origin, room_size);
}

DemoSystem::~DemoSystem() {

}

void DemoSystem::startDemo() {
	if (render_mode_) {
		initWindow();
		initSpheres();
		mainLoop();
	} else {
		testPerformance();
	}
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
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->framebufferSizeCallback(width, height);
	};
	auto mouse_callback_func = [](GLFWwindow* window, double xpos, double ypos) {
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->mouseCallback(xpos, ypos);
	};
	auto scroll_callback_func = [](GLFWwindow* window, double xoffset, double yoffset) {
		static_cast<DemoSystem*>(glfwGetWindowUserPointer(window))->scrollCallback(xoffset, yoffset);
	};

	glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback_func);
	glfwSetCursorPosCallback(window_, mouse_callback_func);
	glfwSetScrollCallback(window_, scroll_callback_func);

	if (immersive_mode_) {
		// tell glfw to capture our mouse
		glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	last_mouse_x_ = WINDOW_WIDTH * 0.5f;
	last_mouse_y_ = WINDOW_HEIGHT * 0.5f;
	first_mouse_ = true;
}

void DemoSystem::initSpheres() {
	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);

	camera_ = new Camera(glm::vec3(0.5f, 0.5f, 0.5f), 5, -20, 225);

	// initialize vertex arrays and buffers for rendering spheres 
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> vertex_normals;
	std::vector<uint> vertex_indices;
	glGenVertexArrays(1, &sphere_VAO_);
	glGenBuffers(1, &sphere_VBO_);
	glGenBuffers(1, &sphere_EBO_);

	for (unsigned int y = 0; y <= VERTICAL_FRAGMENT_NUM; ++y) {
		for (unsigned int x = 0; x <= HORIZONTAL_FRAGMENT_NUM; ++x) {
			float xSegment = (float)x / (float)HORIZONTAL_FRAGMENT_NUM;
			float ySegment = (float)y / (float)VERTICAL_FRAGMENT_NUM;
			float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
			float yPos = std::cos(ySegment * PI);
			float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
			vertices.push_back(glm::vec3(xPos, yPos, zPos));
			vertex_normals.push_back(glm::vec3(xPos, yPos, zPos));
		}
	}
	bool odd_row = false;
	for (unsigned int y = 0; y < VERTICAL_FRAGMENT_NUM; ++y) {
		if (!odd_row) {
			for (unsigned int x = 0; x <= HORIZONTAL_FRAGMENT_NUM; ++x) {
				vertex_indices.push_back(y * (HORIZONTAL_FRAGMENT_NUM + 1) + x);
				vertex_indices.push_back((y + 1) * (HORIZONTAL_FRAGMENT_NUM + 1) + x);
			}
		} else {
			for (int x = HORIZONTAL_FRAGMENT_NUM; x >= 0; --x) {
				vertex_indices.push_back((y + 1) * (HORIZONTAL_FRAGMENT_NUM + 1) + x);
				vertex_indices.push_back(y * (HORIZONTAL_FRAGMENT_NUM + 1) + x);
			}
		}
		odd_row = !odd_row;
	}

	sphere_index_count_ = vertex_indices.size();
	std::vector<float> sphere_data;
	for (std::size_t i = 0; i < vertices.size(); ++i) {
		sphere_data.push_back(vertices[i].x);
		sphere_data.push_back(vertices[i].y);
		sphere_data.push_back(vertices[i].z);
		if (vertex_normals.size() > 0) {
			sphere_data.push_back(vertex_normals[i].x);
			sphere_data.push_back(vertex_normals[i].y);
			sphere_data.push_back(vertex_normals[i].z);
		}
	}

	glBindVertexArray(sphere_VAO_);
	glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO_);
	glBufferData(GL_ARRAY_BUFFER, sphere_data.size() * sizeof(float), &sphere_data[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertex_indices.size() * sizeof(unsigned int), &vertex_indices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// initialize vertex arrays and buffers for rendering background (walls and floor)
	float face_vertices[] = {
		// vertices           // normals          // texture coords
		 1.0f, 1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 1.0f, // top right
		 1.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f, // bottom right
		 0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
		 0.0f, 1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 1.0f  // top left 
	};
	uint face_indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};

	glGenVertexArrays(1, &background_VAO_);
	glGenBuffers(1, &background_VBO_);
	glGenBuffers(1, &background_EBO_);

	glBindVertexArray(background_VAO_);

	glBindBuffer(GL_ARRAY_BUFFER, background_VBO_);
	glBufferData(GL_ARRAY_BUFFER, sizeof(face_vertices), face_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, background_EBO_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(face_indices), face_indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	/*
   Here we set all the uniforms for the 5/6 types of lights we have. We have to set them manually and index
   the proper PointLight struct in the array to set each uniform variable. This can be done more code-friendly
   by defining light types as classes and set their values in there, or by using a more efficient uniform approach
   by using 'Uniform buffer objects', but that is something we'll discuss in the 'Advanced GLSL' tutorial.
*/

	// initialize model matrices for rendering background
	model_matrices_ = { };
	glm::mat4 model_base = glm::mat4(1.0f);
	model_base = glm::translate(model_base, origin_);
	
	glm::mat4 model_right = model_base;
	glm::mat4 model_left = glm::translate(model_base, glm::vec3(0.0f, 0.0f, room_size_.z));
	glm::mat4 model_bottom = model_left;
	//model_right = glm::scale(model_right, glm::vec3(1.0f));
	model_left = glm::rotate(model_left, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	model_bottom = glm::rotate(model_bottom, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	model_matrices_.push_back(model_right);
	model_matrices_.push_back(model_left);
	model_matrices_.push_back(model_bottom);

	// initialize pointlight positions
	pointlight_positions_ = {
		glm::vec3(0.5f,  1.2f,  0.5f),
	};

	// load textures and bind
	wall_texture_ = loadTexture("../resources/textures/brickwall.jpg");
	floor_texture_ = loadTexture("../resources/textures/wood.png");
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, wall_texture_);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, floor_texture_);

	// initialize shaders (background and sphere)
	background_shader_ = initShader("../shaderPrograms/background.vs", "../shaderPrograms/phong.fs", wall_texture_, false);
	sphere_shader_ = initShader("../shaderPrograms/sphere.vs", "../shaderPrograms/phong.fs", 0, true);
}

Shader *DemoSystem::initShader(char const * vs_path, char const * fs_path, uint texture_id, bool has_specular_map)
{
	Shader *shader = new Shader();
	// shader configuration
	shader->loadProgram(vs_path, fs_path);
	shader->use();

	// point light 1
	shader->setVec3("pointLights[0].position", pointlight_positions_[0]);
	shader->setVec3("pointLights[0].ambient", 0.2f, 0.2f, 0.2f);
	shader->setVec3("pointLights[0].diffuse", 0.6f, 0.6f, 0.6f);
	shader->setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
	shader->setFloat("pointLights[0].constant", 1.0f);
	shader->setFloat("pointLights[0].linear", 0.09f);// 0.09);
	shader->setFloat("pointLights[0].quadratic", 0.032f);// 0.032);

	// directional light
	//shader->setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
	shader->setVec3("dirLight.direction", -1.0f, -1.0f, -1.0f);
	shader->setVec3("dirLight.ambient", 0.2f, 0.2f, 0.2f);
	shader->setVec3("dirLight.diffuse", 0.3f, 0.3f, 0.3f);
	shader->setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);

	bool has_texture = texture_id != 0;
	std::cout << std::endl << has_texture;
	shader->setBool("HasTexture", has_texture);
	shader->setBool("HasSpecularMap", has_specular_map);
	shader->setBool("HasSpotLight", use_spotlight_);

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
		shader->setInt("material.diffuseMap", texture_id - 1);
	}

	return shader;
}

void DemoSystem::mainLoop() {
	while (!glfwWindowShouldClose(window_)) {
		float time_start = glfwGetTime();

		// handle keyboard input
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
	glDeleteVertexArrays(1, &background_VAO_);
	glDeleteBuffers(1, &sphere_VBO_);
	glDeleteBuffers(1, &sphere_EBO_);
	glDeleteBuffers(1, &background_VBO_);
	glDeleteBuffers(1, &background_EBO_);

	glfwTerminate();
}

void DemoSystem::renderBackground() {
	// render two walls and the floor
	background_shader_->use();
	updateViewpoint(background_shader_);
	glBindVertexArray(background_VAO_);

	background_shader_->setInt("material.diffuseMap", wall_texture_ - 1);
	background_shader_->setMat4("model", model_matrices_[0]);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
	background_shader_->setMat4("model", model_matrices_[1]);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);

	// switch texture
	background_shader_->setInt("material.diffuseMap", floor_texture_ - 1);
	background_shader_->setMat4("model", model_matrices_[2]);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void *)0);
}

void DemoSystem::renderSpheres() {
	sphere_shader_->use();
	updateViewpoint(sphere_shader_);

	// get updated vertices from engine
	engine_->update(simulation_timestep_);
	float* updated_pos = engine_->outputPos();
	uint *type = engine_->getSphereType();

	glBindVertexArray(sphere_VAO_);
	for (uint i = 0; i < sphere_num_; ++i) {
		// calculate the model matrix for each sphere
		glm::mat4 model = glm::mat4(1.0f);
		uint i_x3 = i * 3;
		model = glm::translate(model, glm::vec3(updated_pos[i_x3], updated_pos[i_x3 + 1], updated_pos[i_x3 + 2]));
		sphere_shader_->setMat4("model", model);

		// set shader attributes according to the sphere type (material and radius)
		Sphere proto = PROTOTYPES[type[i]];
		sphere_shader_->setFloat("radius", proto.radius);
		sphere_shader_->setFloat("material.shininess", proto.shininess);
		sphere_shader_->setVec3("material.ambient", proto.ambient);
		sphere_shader_->setVec3("material.diffuse", proto.diffuse);
		sphere_shader_->setVec3("material.specular", proto.specular);

		glDrawElements(GL_TRIANGLE_STRIP, sphere_index_count_, GL_UNSIGNED_INT, 0);
	}
}

void DemoSystem::testPerformance(uint test_iters) {
	LARGE_INTEGER frequency, startCount, stopCount;
	BOOL ret;

	// us level timer
	ret = QueryPerformanceFrequency(&frequency);
	if (ret) {
		ret = QueryPerformanceCounter(&startCount);
	}
	for (uint i = 0; i < test_iters; ++i) {
		engine_->update(0.1f);
	}
	if (ret) {
		QueryPerformanceCounter(&stopCount);
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
	glm::mat4 view = camera_->getViewMatrix();
	shader->setMat4("projection", projection);
	shader->setMat4("view", view);
}

uint DemoSystem::loadTexture(char const * path) {
	unsigned int textureID;
	glGenTextures(1, &textureID);

	int width, height, nrComponents;
	unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
	if (data) {
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
	} else {
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
		camera_->processKeyboard(FORWARD, loop_duration_);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera_->processKeyboard(BACKWARD, loop_duration_);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera_->processKeyboard(LEFT, loop_duration_);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera_->processKeyboard(RIGHT, loop_duration_);
}

void DemoSystem::framebufferSizeCallback(int width, int height) {
	// make sure the viewport matches the new window dimensions; 
	// width and height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

void DemoSystem::mouseCallback(double pos_x, double pos_y) {
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

	camera_->processMouseMovement(xoffset, yoffset);
}

void DemoSystem::scrollCallback(double xoffset, double yoffset) {
	camera_->processMouseScroll(yoffset);
}