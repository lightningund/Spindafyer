#include <iostream>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernel.h"

// Wrapper for an stbi image
struct Img {
	u8_t* img_data;
	int width;
	int height;
	int bpp;

	Img(std::string filename) {
		img_data = stbi_load(filename.c_str(), &width, &height, &bpp, 0);
	}

	~Img() {
		stbi_image_free(img_data);
	}

	u8_t operator[] (size_t idx) {
		assert(idx < width * height * bpp);
		return img_data[idx];
	}
};

struct Spot {
	std::array<float, k_len> kernel{};

public:
	Spot(std::string filename) {
		Img img{filename};

		assert(img.width == k_size);
		assert(img.height == k_size);

		// Turn the spot image into a kernel
		for (int i = 0; i < img.width * img.height; ++i) {
			kernel[i] = img[i * img.bpp] == 0 ? 1 : -1;
			kernel[i] /= k_len;
		}
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

int main() {
	Spot spot_a{"res/spot_1.png"};

	// Copy the kernel data over to the device
	cudaMemcpyToSymbol(kernel, spot_a.kernel.data(), k_len * sizeof(float), 0, cudaMemcpyHostToDevice);

	int width, height, bpp;

	u8_t* test_img = stbi_load("res/test.png", &width, &height, &bpp, 0);

	size_t img_size = width * height * sizeof(u8_t);

	// Shrink down the image to be one byte per pixel
	for (int i = 0; i < width * height; ++i) {
		test_img[i] = test_img[i * bpp];
	}

	std::cout << img_size << ", " << width << ", " << height << ", " << bpp << "\n";

	Managed<u8_t> dev_img{img_size};
	Managed<u8_t> result_img{img_size};
	cudaMemcpy(dev_img.raw, test_img, img_size, cudaMemcpyHostToDevice);

	stbi_image_free(test_img);

	Managed<float> temp_arr{img_size * sizeof(float)};

	for (int i = 0; i < (width / 16) - 1; ++i) {
		for (int j = 0; j < (height / 16) - 1; ++j) {
			// convolve<<<1, dim3{16, 16, 1}>>>(device_img, i * 16, j * 16, width, result_img);
			conv_atomic<<<dim3{k_size, k_size, 1}, dim3{16, 16, 1}>>>(dev_img.raw, i * 16, j * 16, width, temp_arr.raw);
		}
	}

	// convolve<<<1, dim3{16, 16, 1}>>>(device_img, kernel, 0, 0, width, result_img);

	cudaDeviceSynchronize();

	convert<<<height, width>>>(temp_arr.raw, result_img.raw, img_size);

	cudaDeviceSynchronize();

	// Debug
	for (int j = 0; j < 1; ++j) {
		for (int i = 0; i < 20; ++i) {
			std::cout << (int)result_img.raw[i + j * width] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

	// Write the convolved image back to file
	stbi_write_png("convolved.png", width, height, 1, result_img.raw, 1 * width);

	return 0;
}