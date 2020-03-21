//optimization homework #4 cs 677 Theodore Jagodits

#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <iostream>

#define DEFAULT_WIDTH 128
#define DEFAULT_HEIGHT 128
#define TILE_SIZE 16

__global__ void unknown_algo(float *inp1, float *inp2, float *result, int width, int height){
	//make shared
	__shared__ int temp_s[TILE_SIZE];
	//get row col idx
	int tx = blockIdx.x * blockDim.x + threadIdx.x, ty = blockIdx.y * blockDim.y + threadIdx.y;
	//check for bounds here
	float temp, k_loop_temp, inpt1_s;
	//store into shared
	//temp_s[threadIdx.x] = inp2[ty*width + tx];
	temp = 0.0f;
	//go until your index
	for(int j = 0; j < tx; j++){
		temp += temp_s[];
	}
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
	
	//malloc device
	float *d_input1 = (float *) malloc(input1_bytes);
	float *d_input2 = (float *) malloc(num_bytes);
	float *d_result = (float *) malloc(num_bytes);
	
	//malloc host
	float *h_input1 = (float *) malloc(input1_bytes);
	float *h_input2 = (float *) malloc(num_bytes);
	float *h_result = (float *) malloc(num_bytes);  
	
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
	int block_size = 128;
	//int grid_size = (int)ceil((float)width/block_size);
	int gridx = (int)ceil((float)width/block_size);
	int gridy = (int)ceil((float)height/block_size);
	dim3 dim_block (block_size, block_size);
	dim3 dim_grid(gridx, gridy);
	
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
	
	//print output
	for(int o = 0; o < width; o++){
        for(int p = 0; p < height; p++){
            //printf("%d ", (int)h_result[o*width + p]);
        }
		//printf("\n");
    }
	printf("time for execution: %lf ms\n", milliseconds);
	
	//free all vars
	free(h_input1);
	free(h_input2);
	free(h_result);
	cudaFree(d_input1);
	cudaFree(d_input2);
	cudaFree(d_result);
	
	return 0;
}