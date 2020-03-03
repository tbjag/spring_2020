#define SOBEL_SIZE 9
#define TILE_SIZE 16
#define MASK_WIDTH 3

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sobel_filter(int *inputM, int *outputM, int width, int height, int thresh){
	__shared__ int local[TILE_SIZE][TILE_SIZE];
	
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	local[threadIdx.x][threadIdx.y] = inputM[idx_x*width + idx_y];
	int gx = 0,gy = 0;
	int magnitude, result;
	// Check within bounds
	if(idx_y + 1< height && idx_x + 1< width && idx_y - 1 >= 0 && idx_x -1 >= 0){
		// find gx within bounds
		//base case left
		if(tx == 0){
			// if bottom left
			if(ty + 1 > blockDim.y){
				gx = 
				gy = 
			}else if(ty == 0){
				int top_left = inputM[(idx_x-1)*width + idx_y -1];
				int bot_left = inputM[(idx_x+1)*width + idx_y -1];
				//here firx
				gx = top_left - local[tx+1][ty-1] + (2*inputM[idx_x*width + idx_y -1]) - (2*local[tx+1][ty]) + bot_left - local[tx+1][ty+1];
				gy = 
			}else{
				int top_left = inputM[(idx_x-1)*width + idx_y -1];
				int bot_left = inputM[(idx_x+1)*width + idx_y -1];
				gx = top_left - local[tx+1][ty-1] + (2*inputM[idx_x*width + idx_y -1]) - (2*local[tx+1][ty]) + bot_left - local[tx+1][ty+1];
				gy = top_left + (2 * local[tx][ty-1]) + local[tx+1][ty-1] - bot_left - 2*local[tx][ty+1] - local[tx+1][ty+1];
			}
			
			
		}
	}
	
	magnitude = gx*gx + gy*gy;
	
	if(magnitude > thresh)
		result = 255;
	else
		result = 0;
	
	outputM[idx_x*width + idx_y] = result;  
}
