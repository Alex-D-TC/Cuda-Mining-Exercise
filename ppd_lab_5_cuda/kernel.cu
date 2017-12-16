#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>

#define ROR(bytes, cnt) ( (bytes >> cnt) | ( bytes << ((sizeof(bytes) * 8) - cnt) ) )
#define ROL(bytes, cnt) ( (bytes << cnt) | ( bytes >> ((sizeof(bytes) * 8) - cnt) ) )
#define CUDA_HANDLE_ERR(err, desc_str) if(err != cudaSuccess) printf("%s: %s\n", desc_str, cudaGetErrorString(err));
#define CUDA_HANDLE_ERR_BOOL_FLAG(err, desc_str, flag) if(err != cudaSuccess) { printf("%s: %s\n", desc_str, cudaGetErrorString(err)); flag = true; }

__host__ __device__ bool increment(uint8_t *bytes, uint32_t bytes_len) {
	bool overflow = true;
	for (int64_t i = bytes_len - 1; overflow && i >= 0; --i) {
		bytes[i] += 1;
		overflow = bytes[i] == 0;
	}
	return overflow;
}

__host__ __device__ uint32_t zero_count(uint8_t *in_b, size_t in_b_len) {

	uint32_t count = 0;
	bool done = false;
	for (size_t i = 0; !done && i < in_b_len; ++i) {
		uint8_t data = in_b[i];
		for (char j = 0; !done && j < 8; ++j) {
			if (((data & (1 << 7)) >> 7) % 2 == 0)
				count += 1;
			else
				done = true;
			data <<= 1;
		}
	}

	return count;
}

__host__ __device__ void hash_256(const uint8_t *in_b, size_t in_b_len, uint8_t *out_b) {
	uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a,
		e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

	// prepare padding (if needed)
	uint64_t to_pad = 32 - in_b_len % 32;
	uint64_t block_cnt = (in_b_len + to_pad) / 32;
	
	uint32_t h0 = a, h1 = b, h2 = c, h3 = d, h4 = e, h5 = f, h6 = g, h7 = h;

	// compute
	for (size_t run = 0; run < block_cnt; ++run) {

		a = h0;
		b = h1;
		c = h2;
		d = h3;
		e = h4;
		f = h5;
		g = h6;
		h = h7;

		size_t step = 0;
		for (size_t block_idx = run * 32; step < 8; ++step, block_idx += 4) {
			uint32_t block = 0;
			uint8_t block_size = sizeof(block);
			size_t overflow_len = 0;

			if (block_idx + 4 > in_b_len)
				overflow_len = block_idx + block_size - in_b_len;

			if (overflow_len > 4)
				overflow_len = 4;

			if (block_idx < in_b_len) {
				/*for (size_t i = 0; i < block_size - overflow_len; ++i) {
					*((uint8_t*)(&block) + i) = *(in_b + block_idx + i);
				}*/
				std::memcpy(&block, in_b + block_idx, block_size - overflow_len);
			}

			uint32_t ta = (((ROL(a, 3) & ~ROR(e, 7)) ^ (~ROL(d, 5) & ROR(e, 7))) ^ ROR(block, 3));
			uint32_t tb = (ROR(block, 11) & ROR(a, 13)) ^ block;

			// compute something with the nums
			h = g;
			g = f;
			f = e;
			e = d + ta;
			d = c;
			c = b;
			b = a;
			a = ta + tb;
		}

		h0 = a;
		h1 = b;
		h2 = c;
		h3 = d;
		h4 = e;
		h5 = f;
		h6 = g;
		h7 = h;
	}

	// write result
	for (size_t i = 0; i < 8; ++i) {

		uint32_t num = h0;
		switch (i) {
		case 1:
			num = h1;
			break;
		case 2:
			num = h2;
			break;
		case 3:
			num = h3;
			break;
		case 4:
			num = h4;
			break;
		case 5:
			num = h5;
			break;
		case 6:
			num = h6;
			break;
		case 7:
			num = h7;
			break;
		}

		for (size_t j = 0; j < 4; ++j) {
			uint8_t res = ((num & (255ull << 24)) >> 24) % 256;
			size_t byte_idx = i * 4 + j;
			out_b[byte_idx] = res;
			num <<= 8;
		}
	}
}

__device__ void print_arr(const uint8_t * arr, size_t arr_len, size_t line_len) {
	for (size_t i = 0; i < arr_len; ++i) {
		printf("%d ", arr[i]);
		if ((i + 1) % line_len == 0)
			printf("\n");
	}
	printf("\n");
}

__global__ void kernel_hash(
	uint8_t *blocks,
	size_t blocks_pitch,
	uint32_t block_len,
	uint32_t nonce_idx,
	uint32_t nonce_len,
	bool *nonce_res,
	bool *done,
	uint8_t *thread_counters,
	size_t thread_counters_pitch,
	uint8_t *thread_hashes,
	size_t thread_hashes_pitch,
	size_t hash_len,
	size_t thread_count,
	uint32_t diff,
	size_t step_count) {
	
	// initialize global variable done
	size_t blockNumInGrid = blockIdx.x + gridDim.x  * blockIdx.y;
	size_t threadsPerBlock = blockDim.x * blockDim.y;
	size_t threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;

	size_t globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	size_t thread_idx = globalThreadNum;
	size_t block_idx = thread_idx * (block_len + blocks_pitch);
	size_t thread_counter_idx = thread_idx * (thread_counters_pitch + nonce_len);
	size_t thread_hashes_idx = thread_idx * (thread_hashes_pitch + hash_len);
	bool overflow = false;

	*done = false;

	// clear the counter
	for (size_t i = 0; i < nonce_len; ++i) {
		thread_counters[thread_counter_idx + i] = 0;
	}

	// set the hash
	for (size_t i = 0; i < hash_len; ++i) {
		thread_hashes[thread_hashes_idx + i] = 255;
	}

	while (!overflow && !(*done)) {

		hash_256(blocks + block_idx, block_len, thread_hashes + thread_hashes_idx);
		//printf("%llu\n", size_t(blocks));
		//__syncthreads();

		uint32_t z_cnt = zero_count(thread_hashes + thread_hashes_idx, hash_len);
		if (z_cnt >= diff) {
			*done = true;
			nonce_res[thread_idx] = true;
		}
		else {
			// increment nonce
			for (size_t i = 0; !overflow && i < thread_count; ++i)
				overflow = increment(blocks + block_idx + nonce_idx, nonce_len);
			if (overflow) {
				*done = true;
			}

			// increment counter
			overflow = increment(thread_counters + thread_counter_idx, nonce_len);
			if (overflow) {
				*done = true;
			}
		}
	}
}

uint8_t* build_random_char_sequence(uint32_t len) {
	uint8_t *data = new uint8_t[len];
	for (uint32_t i = 0; i < len; ++i) {
		data[i] = rand() % 256;
	}
	return data;
}

void test_hash_256(size_t run_cnt) {

	uint8_t hash[32]{ 0 };
	uint8_t *bytes = build_random_char_sequence(15);
	std::memset(bytes + 3, 0, 5);
	
	while (run_cnt--) {
		hash_256(bytes, 15, hash);
		increment(bytes + 3, 5);
	}
}

int main() {
	// host data
	const size_t step_count = 100;
	const uint32_t block_count = 1;
	const uint32_t threads_per_block = 500;
	const uint32_t thread_count = block_count * threads_per_block;
	const uint32_t block_len = 26;
	const uint32_t nonce_idx = 5;
	const uint32_t nonce_len = 10;
	const uint32_t diff = 16;
	const uint8_t hash_len = 32;

	bool *nonce_res;
	uint8_t *blocks;
	int64_t correct_idx = -1;
	uint8_t counter[nonce_len];
	uint8_t *block = build_random_char_sequence(block_len);

	// result nonce
	uint8_t nonce[nonce_len]{ 0 };
	uint8_t *counters;

	// device data
	uint8_t *d_blocks;
	bool *d_nonce_res;
	bool *d_done;
	uint8_t *d_thread_counters;
	uint8_t *d_thread_hashes;

	// alloc space
	nonce_res = new bool[thread_count];
	for (size_t i = 0; i < thread_count; ++i)
		nonce_res[i] = false;

	CUDA_HANDLE_ERR(cudaMalloc(&d_nonce_res, thread_count * sizeof(bool)), "Error on malloc for d_nonce_res");
	
	blocks = new uint8_t[block_len * thread_count];
	CUDA_HANDLE_ERR(cudaMalloc(&d_blocks, thread_count * block_len), "Error on malloc for d_blocks");
	
	CUDA_HANDLE_ERR(cudaMalloc(&d_done, sizeof(bool)), "Error on malloc for d_done");
	
	counters = new uint8_t[thread_count * nonce_len];
	CUDA_HANDLE_ERR(cudaMalloc(&d_thread_counters, thread_count * nonce_len), "Error on malloc for d_thread_counters");
	
	CUDA_HANDLE_ERR(cudaMalloc(&d_thread_hashes, thread_count * hash_len), "Error on malloc for d_thread_hashes");

	for (size_t i = 0; i < thread_count; ++i)
		nonce_res[i] = false;

	std::memset(counter, 0, nonce_len);
	// prep block data
	for (size_t i = 0; i < thread_count; ++i) {
		if (i)
			increment(counter, nonce_len);
	
		std::memcpy(blocks + (i * block_len), block, block_len);
		std::memcpy(blocks + (i * block_len) + nonce_idx, counter, nonce_len);
	}

	CUDA_HANDLE_ERR(cudaMemcpy(d_blocks, blocks, thread_count * block_len, cudaMemcpyHostToDevice), "Error on memcpy for blocks to d_blocks");
	CUDA_HANDLE_ERR(cudaMemcpy(d_nonce_res, nonce_res, thread_count * sizeof(bool), cudaMemcpyHostToDevice), "Error on memcpy for nonce_res to d_nonce_res");

	bool error_occured = false;

	do {
		kernel_hash<<<block_count, threads_per_block>>>(
			d_blocks, 0, block_len, nonce_idx, nonce_len, 
			d_nonce_res, 
			d_done, 
			d_thread_counters, 0, 
			d_thread_hashes, 0, hash_len, 
			thread_count, diff,
			step_count);

		CUDA_HANDLE_ERR_BOOL_FLAG(cudaGetLastError(), "Sync kernel error", error_occured);
		CUDA_HANDLE_ERR_BOOL_FLAG(cudaDeviceSynchronize(), "Async kernel error", error_occured);
		CUDA_HANDLE_ERR_BOOL_FLAG(cudaMemcpy(nonce_res, d_nonce_res, thread_count * sizeof(bool), cudaMemcpyDeviceToHost), "Error on getting nonce_res for data", error_occured);
		CUDA_HANDLE_ERR_BOOL_FLAG(cudaMemcpy(counters, d_thread_counters, thread_count * nonce_len, cudaMemcpyDeviceToHost), "Error on getting counters from device", error_occured);
		
		for (uint32_t i = 0; correct_idx == -1 && i < thread_count; ++i)
			if (nonce_res[i])
				correct_idx = i;

	} while (!error_occured && correct_idx == -1);

	if (correct_idx == -1)
		goto cleanup;

	// do something with the nonce
	std::cout << correct_idx << "\n";
	for (uint8_t step_counter[nonce_len]{ 0 }; memcmp(step_counter, counters + correct_idx * nonce_len, nonce_len) != 0; increment(step_counter, nonce_len)) {
		for (size_t i = 0; i < thread_count; ++i)
			increment(nonce, nonce_len);
	}

	// find the starting nonce of the 'winning' thread
	while (correct_idx--)
		increment(nonce, nonce_len);

	// print the nonce
	for (size_t i = 0; i < nonce_len; ++i)
		std::cout << int(nonce[i]) << " ";
	std::cout << "\n";

	// print the hash
	for (size_t i = 0; i < nonce_len; ++i)
		block[nonce_idx + i] = nonce[i];

	uint8_t hash_res[32]{ 0 };
	hash_256(block, block_len, hash_res);
	
	for (size_t i = 0; i < hash_len; ++i)
		std::cout << int(hash_res[i]) << " ";
	std::cout << "\n";

cleanup:

	cudaFree(d_nonce_res);
	cudaFree(d_blocks);
	cudaFree(d_done);
	cudaFree(d_thread_counters);
	cudaFree(d_thread_hashes);

	delete[] counters;
	delete[] nonce_res;
	delete[] blocks;
	delete[] block;
	return 0;
}
