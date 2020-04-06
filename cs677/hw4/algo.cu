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
#define GPU_SPEED_LO_LIMIT 1

//add constant memory 
__constant__ float c_inp[MAX_CONST];

//tiling complete
__global__ void unknown_algo_inp2_chunk(float *result, float *inp2, int row, int col, int num_tiles, int chunk){
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
		if(pos_x < col && ty < row){
			temp_shared_2[threadIdx.y * TILE_SIZE + threadIdx.x] = inp2[ty*row + pos_x];
		}
		__syncthreads();
		for(unsigned int tile = 0; tile < TILE_SIZE; tile++){
			if(curr <= tx){
				temp += temp_shared_2[threadIdx.y * TILE_SIZE + tile];
			}
			curr ++;
		}
	}
	if(ty < row && tx < col){
		result[ty*row + tx] = temp;
	}
}

//tiling complete
__global__ void unknown_algo_inp1_chunk(float *result, int row, int col, int num_tiles, int stride, int chunk){
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	
	float temp, local;
	temp = 0.0f;
	if(ty < row && tx < col){
		local = c_inp[ty*row +tx];
		result[ty*row + tx] = temp;
	}
	
	for(int i = 0; i < col; i++){
		temp += local * c_inp[stride + i];
	}
	
	if(ty < row && tx < col){
		result[ty*row + tx] = temp;
	}
}

//tiling complete
__global__ void unknown_algo_inp2_no_chunk(float *result, int row, int col, int num_tiles){
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
		if(pos_x < col && ty < row){
			temp_shared_2[threadIdx.y * TILE_SIZE + threadIdx.x] = c_inp[ty*row + pos_x];
		}
		__syncthreads();
		for(unsigned int tile = 0; tile < TILE_SIZE; tile++){
			if(curr <= tx){
				temp += temp_shared_2[threadIdx.y * TILE_SIZE + tile];
			}
			curr ++;
		}
	}
	if(ty < row && tx < col){
		result[ty*row + tx] = temp;
	}
}

//tiling complete no point of tiling here
__global__ void unknown_algo_inp1_no_chunk(float *result, int row, int col, int num_tiles, int stride){
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	
	float temp, local;
	temp = 0.0f;
	if(ty < row && tx < col){
		local = c_inp[ty*row +tx];
		result[ty*row + tx] = temp;
	}
	
	for(int i = 0; i < col; i++){
		temp += local * c_inp[stride + i];
	}
	
	if(ty < row && tx < col){
		result[ty*row + tx] = temp;
	}
}

int main( int argc, char **argv ){
	int row = DEFAULT_ROW;
	int col = DEFAULT_COL;
	if(argc == 3){
		row = atoi(argv[1]);
		col = atoi(argv[2]);
	}
	int num_bytes = row* col * sizeof(float);
	int stride = row * col;
	
	float *h_temp = (float *) calloc(col, sizeof(float)); //store temp results
	float *d_temp = (float *) calloc(col, sizeof(float)); //store temp results
	float *d_result = (float *) malloc(num_bytes);
	float *h_input = (float *) malloc(num_bytes);
	float *h_result = (float *) malloc(num_bytes);
	
	//cudaMalloc(&d_input2, num_bytes);
	cudaMalloc(&d_result, num_bytes);
	cudaMalloc(&d_temp, col * sizeof(float));
	
	//put in data
	for(int o = 0; o < row; o++){
        for(int p = 0; p < col; p++){
            h_input[row * o + p] = 1.0f;
        }
    }
	// add input 1 to input
	for(int p = 0; p < col; p++){
        h_input[stride + p] = 1.0f;
    }
	
	//declare block and grid size for kernel
	int block_size = TILE_SIZE;
	//make grids x y
	int grid_x = (int)ceil((float)col/block_size);
	int grid_y = (int)ceil((float)row/block_size);
	dim3 dim_grid (grid_x, grid_y);
	dim3 dim_block (block_size, block_size);
	
	//run different configs depending on size
	if(col * row + col> MAX_CONST){ //greater than limit of const mem 
	/*
		float *temp_chunk = (float*) calloc(MAX_CONST,sizeof(float));
		for(int chunk = 0; chunk < (int)ceil((float)(row*col)/MAX_CONST); chunk++){
			for(int i = 0; i < MAX_CONST; i++){
				if(i<stride)
					temp_chunk[i] = h_input[chunk*MAX_CONST + i];
				else
					break;
			}
			cudaMemcpyToSymbol(c_inp, temp_chunk, MAX_CONST);
			unknown_algo_inp2_chunk<<< dim_grid, dim_block >>> (d_result, d_temp, row, col, num_tiles, chunk);
		}
		for(int chunk = 0; chunk < (int)ceil((float)(row*col)/MAX_CONST); chunk++){
			unknown_algo_inp1<<< dim_grid, dim_block >>> (d_result, row, col, 0);
		}
		free(temp_chunk);
		*/
		
	} else if(col * row > GPU_SPEED_LO_LIMIT){//check
		cudaMemcpyToSymbol(c_inp, h_input, num_bytes);
		unknown_algo_inp2_no_chunk<<< dim_grid, dim_block >>> (d_result, row, col, grid_x);
		unknown_algo_inp1_no_chunk<<< dim_grid, dim_block >>> (d_result, row, col, grid_x, stride);
	} else{ // run one kernel
		//combined_algo<<< dim_grid, dim_block >>> (d_result, row, col, 0);
	}
	
	// Copy result back to host
	cudaMemcpy(h_result, d_result, num_bytes, cudaMemcpyDeviceToHost);
	
	//print error check
	for(int i = 0; i < row; i++){
		printf("%d. ", i);
		for(int j = 0; j < col; j++){
			printf("%d ", (int)h_result[i*row + j]);
		}
		printf("\n");
	}
	
	free(h_input);
	free(h_result);
	
	cudaFree(d_result);
	cudaFree(d_temp);
	return 0;
}