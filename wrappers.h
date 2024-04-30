#include <string>
#include <assert.h>
#include <cstdint>

#include "stb_image.h"
#include "stb_image_write.h"

// Wrapper for an stbi image
struct Img {
	uint8_t* img_data;
	int width;
	int height;
	int bpp;

	Img(std::string filename) {
		img_data = stbi_load(filename.c_str(), &width, &height, &bpp, 0);
	}

	~Img() {
		stbi_image_free(img_data);
	}

	uint8_t operator[] (size_t idx) {
		assert(idx < width * height * bpp);
		return img_data[idx];
	}
};

// Wrapper for managed memory objects
template <typename T>
struct Managed {
	T* raw;

	// Takes the size as a number of bytes
	Managed(size_t size) {
		cudaMallocManaged(&raw, size);
	}

	~Managed() {
		cudaFree(raw);
	}
};