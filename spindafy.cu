#include <iostream>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using img_ptr = const uint8_t* const;

constexpr int k_size = 13;
constexpr int k_len = k_size * k_size;

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

	Managed(size_t size) {
		cudaMallocManaged(&raw, size);
	}

	~Managed() {
		cudaFree(raw);
	}
};

__device__ float kernel[k_len];

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
	Spot spot_a{"res/spot_1.png"};

	// Copy the kernel data over to the device
	cudaMemcpyToSymbol(kernel, spot_a.kernel.data(), k_len * sizeof(float), 0, cudaMemcpyHostToDevice);

	int width, height, bpp;

	uint8_t* test_img = stbi_load("res/test.png", &width, &height, &bpp, 0);

	size_t img_size = width * height * sizeof(uint8_t);

	// Shrink down the image to be one byte per pixel
	for (int i = 0; i < width * height; ++i) {
		test_img[i] = test_img[i * bpp];
	}

	std::cout << img_size << ", " << width << ", " << height << ", " << bpp << "\n";

	Managed<uint8_t> dev_img{img_size};
	Managed<uint8_t> result_img{img_size};
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