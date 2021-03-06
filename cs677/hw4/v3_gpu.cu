//optimization homework #4 cs 677 Theodore Jagodits

#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <iostream>

#define DEFAULT_SIZE 128
#define DEFAULT_WIDTH 128
#define DEFAULT_HEIGHT 128
#define TILE_SIZE 16

__global__ void unknown_algo(float *inp1, float *inp2, float *result, int width, int height){
	// make shared
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float temp, k_loop_temp, inpt1_s;
	temp = 0.0f;
	for(int j = 0; j < height; j++){
		temp += inp2[id * width + j];
		result[id * width + j] = temp;
		k_loop_temp = 0.0f;
		inpt1_s = inp1[j];
		for(int k = 0; k < height; k++){
			k_loop_temp += inpt1_s * inp1[k];
		}
		//speed up here
		result[id * width + j] += k_loop_temp;
	}
}

int main( int argc, char **argv ){
	int size = DEFAULT_SIZE;
	int width = DEFAULT_WIDTH;
	int height = DEFAULT_HEIGHT;
	if(argc == 3){
		//size = atoi(argv[1]);
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
	int grid_size = (int)ceil((float)width/block_size);
	
	//start timer 
	cudaEventRecord(start);
	
	//run kernel
	unknown_algo<<< grid_size, block_size >>> (d_input1, d_input2, d_result, width, height);
	
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
            printf("%d ", (int)h_result[o*width + p]);
        }
		printf("\n");
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