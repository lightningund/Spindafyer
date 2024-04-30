#pragma once

using u8_t = unsigned char;
using img_ptr = const u8_t* const;

constexpr int k_size = 13;
constexpr int k_len = k_size * k_size;

__device__ float kernel[k_len];

__global__ void conv_atomic(img_ptr img, const int x, const int y, const int iw, float* temp_arr);
__global__ void convert(float* input, u8_t* output, int n);
__global__ void convolve(img_ptr img, const int x, const int y, const int iw, u8_t* convolved_img);
__global__ void get_spinda_pid(img_ptr img, const int x, const int y, const unsigned int* pids);