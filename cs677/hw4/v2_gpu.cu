//optimization homework #4 cs 677 Theodore Jagodits

#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <iostream>

#define DEFAULT_SIZE 128
#define TILE_SIZE 16

__global__ void unknown_algo(float *inp1, float *inp2, float *result, int size){
	// make shared
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float temp = 0.0f;
	for(int j = 0; j < size; j++){
		temp += inp2[id * size + j];
		result[id * size + j] = temp;
		for(int k = 0; k < size; k++){
			//shared input 1 here
			result[id * size + j] += inp1[j] * inp1[k];
		}
	}
}

int main( int argc, char **argv ){
	int size = DEFAULT_SIZE;
	if(argc == 2){
		size = atoi(argv[1]);
	}
	
	//create vars
	int input1_bytes = size * sizeof(float);
	int num_bytes = size * size * sizeof(float);
	
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
	for(int o = 0; o < size; o++){
        h_input1[o] = 1;
        for(int p = 0; p < size; p++){
            h_input2[size * o + p] = 1;
        }
    }

	//copy over memory
	cudaMemcpy(d_input1, h_input1, input1_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, h_input2, num_bytes, cudaMemcpyHostToDevice);
	
	//declare block and grid size for kernel
	int block_size = 128;
	int grid_size = (int)ceil((float)size/block_size);
	
	//start timer 
	cudaEventRecord(start);
	
	//run kernel
	unknown_algo<<< grid_size, block_size >>> (d_input1, d_input2, d_result, size);
	
	//end timer
	cudaEventRecord(stop);
	
	// Copy result back to host
	cudaMemcpy(h_result, d_result, num_bytes, cudaMemcpyDeviceToHost);
	
	//synchronize https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	//print output
	for(int o = 0; o < size; o++){
        for(int p = 0; p < size; p++){
            printf("%d ", (int)h_result[o*size + p]);
        }
		printf("\n");
    }
	printf("time for execution: %lf ms\n", milliseconds);
	
	//free all vars
	//free(d_input1);
	//free(d_input2);
	//free(d_result);
	free(h_input1);
	free(h_input2);
	free(h_result);
	cudaFree(d_input1);
	cudaFree(d_input2);
	cudaFree(d_result);
	
	return 0;
}