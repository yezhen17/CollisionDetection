/*
 * Entry point
 */

#include <iostream>
#include <stb_image.h>
#define STB_IMAGE_IMPLEMENTATION

#include "global.h"
#include "demoSystem.h"

extern "C" void cudaInit(int argc, char **argv);

int main(int argc, char **argv) {
	uint sphere_num = SPHERE_NUM_DEFAULT;
	bool render_mode = RENDER_MODE_DEFAULT;
	bool gpu_mode = GPU_MODE_DEFAULT;

#if false // for debugging
	char input;
	bool skip = false;
	std::cout << "Use default setting? Press [y] to use and any other key elsewise." << std::endl;
	std::cin >> input;
	if (input == 'y' || input == 'Y') {
		std::cout << "Using default setting." << std::endl;
		skip = true;
	}
	if (!skip) {
		std::cout << "Use render mode? Press [y] to use and any other key elsewise." << std::endl;
		std::cin >> input;
		if (input == 'y' || input == 'Y') {
			std::cout << "Using render mode." << std::endl;
			render_mode = true;
		}
		else {
			std::cout << "Using performance test mode." << std::endl;
			render_mode = false;
		}
		std::cout << "Use GPU? Press [y] to use and any other key elsewise." << std::endl;
		std::cin >> input;
		if (input == 'y' || input == 'Y') {
			std::cout << "Using GPU." << std::endl;
			gpu_mode = true;
		} 
		else {
			std::cout << "Using CPU." << std::endl;
			gpu_mode = false;
		}
			
		std::cout << "How many spheres would you want? Enter a number please." << std::endl;
		uint num;
		std::cin >> num;
		if (num >= 0 && num < 32769) {
			std::cout << "Valid number." << std::endl;
			sphere_num = num;
		}
		else {
			std::cout << "Invalid number! Exiting..." << std::endl;
		}
	}
#endif

	if (gpu_mode) {
		cudaInit(argc, argv);
	}
	std::cout << "************************************" << std::endl;
	std::cout << "******** Starting Demo System *********" << std::endl;
	DemoSystem* demo_system = new DemoSystem(sphere_num, render_mode, gpu_mode);
	demo_system->startDemo();
	delete demo_system;
	std::cout << "********* Exiting Demo System *********" << std::endl;
	std::cout << "************************************" << std::endl;
	system("pause");
	return 0;
}
