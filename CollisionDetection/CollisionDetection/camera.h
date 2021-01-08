/*
 * This file defines a camera class which always points to the origin
 * reference: https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/7.3.camera_mouse_zoom/camera_mouse_zoom.cpp
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float ZOOM = 45.0f;


// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera {
public:
	Camera(float distance, float pitch, float yaw): 
		world_up_(glm::vec3(0.0f, 1.0f, 0.0f)), 
		view_shift_speed_(SPEED),
		zoom_(ZOOM) {
		pitch_ = pitch;
		yaw_ = yaw;
		distance_ = distance;
		updateCameraVectors();
	}

	// returns the view matrix calculated using Euler Angles and the LookAt Matrix
	glm::mat4 getViewMatrix() {
		return glm::lookAt(camera_pos_, glm::vec3(0.0f, 0.0f, 0.0f), camera_up_);
	}

	float getZoom() {
		return zoom_;
	}

	glm::vec3 getCameraPos() {
		return camera_pos_;
	}

	glm::vec3 getCameraFront() {
		return camera_front_;
	}

	// processes input received from any keyboard-like input system. 
	// Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void processKeyboard(Camera_Movement direction, float deltaTime) {
		float velocity = view_shift_speed_ * deltaTime;
		if (direction == FORWARD)
			pitch_ -= glm::degrees(velocity / distance_);
		if (direction == BACKWARD)
			pitch_ += glm::degrees(velocity / distance_);
		if (direction == LEFT)
			yaw_ = yaw_ + glm::degrees(velocity / distance_);
		if (direction == RIGHT)
			yaw_ = yaw_ - glm::degrees(velocity / distance_);
		if (pitch_ > 89.0f)
			pitch_ = 89.0f;
		if (pitch_ < -89.0f)
			pitch_ = -89.0f;

		updateCameraVectors();
	}

	// processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true) {

	}

	// processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void processMouseScroll(float yoffset) {
		zoom_ -= (float)yoffset;
		if (zoom_ < 1.0f)
			zoom_ = 1.0f;
		if (zoom_ > 45.0f)
			zoom_ = 45.0f;
	}

protected:
	// camera Attributes
	glm::vec3 camera_pos_;
	glm::vec3 camera_front_;
	glm::vec3 camera_up_;
	glm::vec3 camera_right_;
	glm::vec3 world_up_;

	// euler Angles
	float yaw_;
	float pitch_;

	// camera distance to origin 
	float distance_;

	// camera options
	float view_shift_speed_;
	float zoom_;

protected:
	// calculates the front vector from the Camera's (updated) Euler Angles
	void updateCameraVectors() {
		glm::vec3 front;
		front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
		front.y = sin(glm::radians(pitch_));
		front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
		camera_front_ = glm::normalize(front);
		camera_pos_ = -camera_front_ * distance_;
		camera_right_ = glm::normalize(glm::cross(camera_front_, world_up_));
		camera_up_ = glm::normalize(glm::cross(camera_right_, camera_front_));
	}
};
#endif
