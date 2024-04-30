#include <iostream>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using img_ptr = const uint8_t* const;

constexpr int k_size = 13;

__device__ float kernel[k_size * k_size];

// The kernel is anchored on the top left corner instead of the center
// threadIdx x and y determine the location in the convolved image
// Each block calculates one pixel in the kernel for all pixels in the convoluted area
// Each thread adds the multiplied value of one pixel in the kernel
__global__
void conv_atomic(img_ptr img, const int x, const int y, const int iw, float* temp_arr) {
	// Make sure it's in the range of possible values for the spot
	if (threadIdx.x >= 16 || threadIdx.y >= 16) return;
	// Make sure it's in the range of the kernel
	if (blockIdx.x >= k_size || blockIdx.y >= k_size) return;

	int conv_x = threadIdx.x + x;
	int conv_y = threadIdx.y + y;
	int conv_idx = conv_x + conv_y * iw;
	int k_idx = blockIdx.x + blockIdx.y * k_size;
	int img_idx = (blockIdx.x + conv_x) + (blockIdx.y + conv_y) * iw;

	atomicAdd(&temp_arr[conv_idx], (img[img_idx] - 128) * kernel[k_idx]);
}

__global__
void convert(float* input, uint8_t* output, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	output[idx] = input[idx];
}

// The kernel is anchored on the top left corner instead of the center
// threadIdx x and y determine the location in the convolved image
// Each instance of this kernel calculates one pixel in the convolved image
__global__
void convolve(img_ptr img, const int x, const int y, const int iw, uint8_t* convolved_img) {
	// Make sure it's in the range of possible values for the spot
	if (threadIdx.x >= 16 || threadIdx.y >= 16) return;

	int conv_x = threadIdx.x + x;
	int conv_y = threadIdx.y + y;
	int conv_idx = conv_x + conv_y * iw;
	float result = 0;
	for (int j = 0; j < k_size; ++j) {
		for (int i = 0; i < k_size; ++i) {
			int k_idx = i + j * k_size;
			int img_idx = (i + conv_x) + (j + conv_y) * iw;
			// printf("x:%d, y:%d, idx:%d\n", i + conv_x, j + conv_y, img_idx);
			result += (img[img_idx] - 128) * kernel[k_idx];
		}
	}

	convolved_img[conv_idx] = result + 128;
	// printf("result:%f\n", result);
}

// Takes in a pointer to the image data and the coordinates of the top left corner
// Returns the PID of the spinda that most closely matches that section
__global__
void get_spinda_pid(img_ptr img, const int x, const int y, const unsigned int* pids) {

}

int main() {
	int width;
	int height;
	int bpp;

    uint8_t* spot_a = stbi_load("res/spot_1.png", &width, &height, &bpp, 0);

	std::array<float, k_size * k_size> host_kernel{};

	// Turn the spot into a kernel
	for (int i = 0; i < width * height; ++i) {
		host_kernel[i] = spot_a[i * bpp] == 0 ? 1 : -1;
		host_kernel[i] /= k_size * k_size;
	}

	cudaMemcpyToSymbol(kernel, host_kernel.data(), k_size * k_size * sizeof(float), 0, cudaMemcpyHostToDevice);

	// Debug
	for (int i = 0; i < 20; ++i) {
		std::cout << (int)spot_a[i] << " ";
	}

	std::cout << "\n" << bpp << "\n";

    stbi_image_free(spot_a);

	uint8_t* test_img = stbi_load("res/test.png", &width, &height, &bpp, 0);

	size_t img_size = width * height * sizeof(uint8_t);

	// Shrink down the image to be one byte per pixel
	for (int i = 0; i < width * height; ++i) {
		test_img[i] = test_img[i * bpp];
	}

	std::cout << img_size << ", " << width << ", " << height << ", " << bpp << "\n";

	uint8_t* device_img;
	cudaMallocManaged(&device_img, img_size);
	cudaMemcpy(device_img, test_img, img_size, cudaMemcpyHostToDevice);

	float* temp_arr;
	cudaMallocManaged(&temp_arr, img_size * sizeof(float));

	stbi_image_free(test_img);

	uint8_t* result_img;
	cudaMallocManaged(&result_img, img_size);

	for (int i = 0; i < (width / 16) - 1; ++i) {
		for (int j = 0; j < (height / 16) - 1; ++j) {
			// convolve<<<1, dim3{16, 16, 1}>>>(device_img, i * 16, j * 16, width, result_img);
			conv_atomic<<<dim3{k_size, k_size, 1}, dim3{16, 16, 1}>>>(device_img, i * 16, j * 16, width, temp_arr);
		}
	}

	// convolve<<<1, dim3{16, 16, 1}>>>(device_img, kernel, 0, 0, width, result_img);

	cudaDeviceSynchronize();

	convert<<<height, width>>>(temp_arr, result_img, img_size);

	cudaDeviceSynchronize();

	for (int j = 0; j < 1; ++j) {
		for (int i = 0; i < 20; ++i) {
			std::cout << (int)result_img[i + j * width] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "\n";

	stbi_write_png("convolved.png", width, height, 1, result_img, 1 * width);

	cudaFree(device_img);
	cudaFree(result_img);

	return 0;
}