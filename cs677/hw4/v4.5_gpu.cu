//optimization homework #4 cs 677 Theodore Jagodits
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <iostream>

#define DEFAULT_WIDTH 128
#define DEFAULT_HEIGHT 128
#define TILE_SIZE 32

__global__ void unknown_algo(float *inp1, float *inp2, float *result, int width, int height){
	//get row col idx
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	//check for bounds here
	if(row < width && col < height){
		//make vars for temp, k_temp, input1[ty]
		float temp = 0.0f, k_loop_t = 0.0f, input1_store = inp1[col];
		//find the first part of the result
		for(int count = 0; count <= col; count++){
			temp += inp2[row*width + count];
		}
		//store result
		result[row*width + col] = 5;
		//find inpt1 of result
			k_loop_t += input1_store * inp1[k];
		for(int k = 0; k < height; k++){
		}
		//add to answer
		result[row*width + col] += k_loop_t;	
	}
}

void cpu_v(float *inp1, float *inp2, float *result, int width, int height, float *temp){
	for(int i = 0 ; i < width; i++){
		temp[i] = 0.0f;
		for(int j = 0; j < height; j++){
			temp[i] += inp2[i*width + j];
			result[i*width + j] = temp[i];
			for(int k = 0; k < height; k++){
				result[i*width + j] += inp1[j] * inp1[k];
			}
		}
	}
}

int compare_res(float *d_res, float *h_res, int width, int height){
	int check = 0;
	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			if(d_res[i*width + j] != h_res[i*width +j]){
				check += 1;
			}
		}
	}
	return check;
}

int main( int argc, char **argv ){
	int width = DEFAULT_WIDTH;
	int height = DEFAULT_HEIGHT;
	if(argc == 3){
		width = atoi(argv[1]);
		height = atoi(argv[2]);
	}
	
	//create vars
	int input1_bytes = height * sizeof(float);
	int num_bytes = width * height * sizeof(float);
	
	//event timers
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	clock_t start_cpu, end_cpu;
	
	//malloc device
	float *d_input1 = (float *) malloc(input1_bytes);
	float *d_input2 = (float *) malloc(num_bytes);
	float *d_result = (float *) malloc(num_bytes);
	
	//malloc host
	float *h_input1 = (float *) malloc(input1_bytes);
	float *h_input2 = (float *) malloc(num_bytes);
	float *h_result = (float *) malloc(num_bytes);
	
	//malloc test
	float *temp = (float *) malloc(width * sizeof(float));
	float *cpu_res = (float *) malloc(num_bytes);
	
	//cuda malloc
	cudaMalloc(&d_input1, input1_bytes);
	cudaMalloc(&d_input2, num_bytes);
	cudaMalloc(&d_result, num_bytes);
	
	//put in data
	for(int o = 0; o < width; o++){
        h_input1[o] = 1;
        for(int p = 0; p < height; p++){
            h_input2[width * o + p] = 1;
        }
    }

	//copy over memory
	cudaMemcpy(d_input1, h_input1, input1_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, h_input2, num_bytes, cudaMemcpyHostToDevice);
	
	//declare block and grid size for kernel
	int block_size = 32;
	//make grids x y
	int grid_x = (int)ceil((float)width/block_size);
	int grid_y = (int)ceil((float)height/block_size);
	dim3 dim_grid (grid_x, grid_y);
	dim3 dim_block (block_size, block_size);
	
	//start timer 
	cudaEventRecord(start);
	
	//run kernel
	unknown_algo<<< dim_grid, dim_block >>> (d_input1, d_input2, d_result, width, height);
	
	//end timer
	cudaEventRecord(stop);

	// Copy result back to host
	cudaMemcpy(h_result, d_result, num_bytes, cudaMemcpyDeviceToHost);
	
	//synchronize https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	//cpu version
	start_cpu = clock();
	cpu_v(h_input1, h_input2, cpu_res, width, height, temp);
	end_cpu = clock();

	int error_count = compare_res(h_result, cpu_res, width, height);
	
	printf("error count: %d\nGPU time for execution: %lf ms\nCPU time for execution: %lf ms\n", error_count, milliseconds, ((float)((end_cpu-start_cpu)*1000))/CLOCKS_PER_SEC);
	printf("gridx: %d, grid y: %d\n", grid_x,grid_y);
	//print error check
	for(int i = 0; i < width; i++){
		printf("%d. ", i);
		for(int j = 0; j < height; j++){
			printf("%d ", (int)h_result[i*width + j]);
		}
		printf("\n");
	}
	for(int i = 0; i < width; i++){
		printf("%d. ", i);
		for(int j = 0; j < height; j++){
			printf("%d ", (int)cpu_res[i*width + j]);
		}
		printf("\n");
	}
	
	//free all vars
	free(h_input1);
	free(h_input2);
	free(h_result);
	free(temp);
	cudaFree(d_input1);
	cudaFree(d_input2);
	cudaFree(d_result);
	
	return 0;
}