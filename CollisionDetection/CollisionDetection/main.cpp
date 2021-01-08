/*
 * Entry point
 */

#include <stb_image.h>
#define STB_IMAGE_IMPLEMENTATION

#include "global.h"
#include "demoSystem.h"

extern "C" void cudaInit(int argc, char **argv);

int main(int argc, char **argv) {
	if (GPU_MODE) {
		cudaInit(argc, argv);
	}
	DemoSystem* demo_system = new DemoSystem();
	demo_system->startDemo();
	delete demo_system;
	return 0;
}
