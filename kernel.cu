#include "kernel.h"
#include <iostream>

__device__ float kernel[k_len];

__constant__ int2 spot_offsets[] = {
	{8, 6},
	{32, 7},
	{14, 24},
	{26, 25}
};

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
	// printf("result:%f\n", input[idx]);
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