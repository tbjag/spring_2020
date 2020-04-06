//optimization homework #4 cs 677 Theodore Jagodits
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <iostream>

#define DEFAULT_ROW 128
#define DEFAULT_COL 128
#define TILE_SIZE 16
#define MAX_CONST 16000

//add constant memory 
__constant__ float c_inp[MAX_CONST];

//tiling complete
__global__ void unknown_algo_inp2(float *inp2, float *result, int row, int col, int num_tiles){
	//add shared memory
	__shared__ float temp_shared_2[TILE_SIZE * TILE_SIZE];
	//get row col idx
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0.0f;
	unsigned int curr = 0, pos_x;
	
	for(unsigned int count = 0; count < num_tiles; count++){
		//find position
		pos_x = count * TILE_SIZE + threadIdx.x;
		//check bounds and load tile
		if(pos_x < col && tx < row){
			temp_shared_2[threadIdx.y * TILE_SIZE + threadIdx.x] = c_inp[ty*row + pos_x];
		}
		__syncthreads();
		for(unsigned int tile = 0; tile < TILE_SIZE; tile++){
			if(curr <= tx){
				temp += temp_shared_2[threadIdx.x * TILE_SIZE + tile];
			}
			curr ++;
		}
	}
	
	if(ty < row && tx < col){
		result[ty*row + tx] = 30;
	}
}

__global__ void unknown_algo_inp1(float *inp1, float *result, int row, int col, int num_tiles){
	//add shared memory
	__shared__ float temp_shared_1[TILE_SIZE][TILE_SIZE];
	//get row col idx
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	float temp, local; 
	unsigned int curr = 0, pos_y;
	//preload local
	if(tx < row && ty < row){
		temp = 0.0f;
		local = inp1[row*tx + ty];
	}
	
	for(unsigned int count = 0; count < num_tiles; count++){
		pos_y = count * TILE_SIZE + threadIdx.y;
		//decrease load by letting one thread do it
		if(pos_y < col && tx < row){
			temp_shared_1[threadIdx.x][threadIdx.y] = inp1[pos_y];
		}
		__syncthreads();
		for(unsigned int tile = 0; tile < TILE_SIZE; tile++){
			if(curr <= ty){
				temp += temp_shared_1[threadIdx.x][tile] * local;
			}
			curr++;
		}
	}

	if(tx < row && ty < col){
		result[tx*row + ty] = temp;
	}
}

void cpu_v(float *inp1, float *inp2, float *result, int row, int col, float *temp){
	for(int i = 0 ; i < row; i++){
		temp[i] = 0.0f;
		for(int j = 0; j < col; j++){
			temp[i] += inp2[i*row + j];
			result[i*row + j] = temp[i];
			for(int k = 0; k < col; k++){
				result[i*row + j] += inp1[j] * inp1[k];
			}
		}
	}
}

int compare_res(float *d_res, float *h_res, int row, int col){
	int check = 0;
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			if(d_res[i*row + j] != h_res[i*row +j]){
				check += 1;
			}
		}
	}
	return check;
}

int main( int argc, char **argv ){
	int row = DEFAULT_ROW;
	int col = DEFAULT_COL;
	if(argc == 3){
		row = atoi(argv[1]);
		col = atoi(argv[2]);
	}
	
	//create vars
	int input1_bytes = col * sizeof(float);
	int num_bytes = row * col * sizeof(float);
	
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
	float *temp = (float *) malloc(row* sizeof(float));
	float *cpu_res = (float *) malloc(num_bytes);
	
	//cuda malloc
	cudaMalloc(&d_input1, input1_bytes);
	cudaMalloc(&d_input2, num_bytes);
	cudaMalloc(&d_result, num_bytes);
	
	//put in data
	for(int o = 0; o < row; o++){
        for(int p = 0; p < col; p++){
            h_input2[row * o + p] = 1.0f;
        }
    }
	for(int i = 0; i < col; i++){
		h_input1[i] = 1.0f;
	}

	//copy over memory
	cudaMemcpy(d_input1, h_input1, input1_bytes, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_input2, h_input2, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_inp, h_input2, num_bytes);
	
	//declare block and grid size for kernel
	int block_size = TILE_SIZE;
	//make grids x y
	int grid_x = (int)ceil((float)row/block_size);
	int grid_y = (int)ceil((float)col/block_size);
	dim3 dim_grid (grid_x, grid_y);
	dim3 dim_block (block_size, block_size);
	
	//start timer 
	cudaEventRecord(start);
	
	//run kernel on inp2 
	unknown_algo_inp2<<< dim_grid, dim_block >>> (d_input2, d_result, row, col, grid_x);
	//inp1
	//unknown_algo_inp1<<< dim_grid, dim_block >>> (d_input1, d_result, row, col, grid_x);
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
	cpu_v(h_input1, h_input2, cpu_res, row, col, temp);
	end_cpu = clock();

	int error_count = compare_res(h_result, cpu_res, row, col);
	
	printf("error count: %d\nGPU time for execution: %lf ms\nCPU time for execution: %lf ms\n", error_count, milliseconds, ((float)((end_cpu-start_cpu)*1000))/CLOCKS_PER_SEC);
	
	printf("gridx: %d gridy: %d\n", grid_x, grid_y);
	//print error check
	for(int i = 0; i < row; i++){
		printf("%d. ", i);
		for(int j = 0; j < col; j++){
			printf("%d ", (int)h_result[i*row + j]);
		}
		printf("\n");
	}
	for(int i = 0; i < row; i++){
		printf("%d. ", i);
		for(int j = 0; j < col; j++){
			printf("%d ", (int)cpu_res[i*row + j]);
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