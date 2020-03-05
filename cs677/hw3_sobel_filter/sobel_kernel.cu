#define TILE_SIZE 16
#define MASK_WIDTH 3

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sobel_filter(int *inputM, int *outputM, int width, int height, int thresh){
	// shared 16x16
	__shared__ int local[TILE_SIZE][TILE_SIZE];
	
	//set up vars, leave 1 px border
	int tx = threadIdx.x; int ty = threadIdx.y;
	//find positions within input array for 14x14
	int row = blockIdx.x * (blockDim.x-2) + tx + 1; 
	int col = blockIdx.y * (blockDim.y-2) + ty + 1;
	
	//correct to the right place in input
	int place = (row-1)*width + col - 1;

	//if in bounds of pic
	if(place < height*width){
		//store into shared
		local[tx][ty] = inputM[place];
		//find inner part of 14x14
		if(tx > 0 && tx < 15 && ty > 0 && ty < 15){
			//no edges or corners
			int top_left = local[tx-1][ty-1], top_right = local[tx+1][ty-1];
			int bot_left = local[tx-1][ty+1], bot_right = local[tx+1][ty+1];
			int gx = top_left - top_right + 2*local[tx-1][ty] - 2*local[tx+1][ty] + bot_left - bot_right;
			int gy = top_left + 2*local[tx][ty-1] + top_right - bot_left - 2*local[tx][ty+1] - bot_right;
			
			// calculate magnitude
			int magnitude = gx*gx + gy*gy;
			int result = 0;
			
			// Check if greater than threshold
			if(magnitude > thresh)
				result = 255;
			
			// store into global
			outputM[place] = result;  
		}
	}
}
