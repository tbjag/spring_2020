#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void maxPerRow(double *values, int *gl_max, int n){
	// Get our global thread ID
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int val = values[id];
	//figure this out
	if(id < n)
		atomicMax(&gl_max[blockIdx.x], val);
}

int main( int argc, char* argv[] ){
	
	// variable size of matrix < 1024
    int col_size = 10, row_size = 1024;
	
	//check that sizes are correct
	if(col_size > 1025 || row_size > 1025){
		printf("size not valid\n");
		return 1;
	}
	
	//how many data points there are row*col
	int data_size = col_size * row_size;
	
	// Host input matrix
    double *h_a;
    //Host output matrix
    int *h_c;
 
    // Device input matrix
    double *d_a;
    //Device output matrix
    int *d_c;
	
	// Size, in bytes, of each matrix
    size_t bytes_input = data_size*sizeof(double);
	size_t bytes_output = col_size*sizeof(int);
	
	// Allocate memory for each matrix on host
    h_a = (double*)malloc(bytes_input);
    h_c = (int*)malloc(bytes_output);
	
	// Allocate memory for each matrix on GPU
    cudaMalloc(&d_a, bytes_input);
    cudaMalloc(&d_c, bytes_output);
	
	int row_id;
    // Initialize matrix on host
	// Simple initialize for now
    for( row_id = 0; row_id < data_size; row_id++ ) {
		h_a[row_id] = 4;
    }
	
	// initialize output to zeroes
	for(row_id = 0; row_id < col_size; row_id++){
		h_c[row_id] = 0;
	}
	//some max values to test
	h_a[4] = 19;
	h_a[16] = 21;
	h_a[98] = 49;
	// Copy host matrices to device
    cudaMemcpy( d_a, h_a, bytes_input, cudaMemcpyHostToDevice);
    cudaMemcpy( d_c, h_c, bytes_output, cudaMemcpyHostToDevice);
	
	// Initialize grid and block
	int blockSize, gridSize;
	
	// Number of threads in each thread block 
    blockSize = 1024;
	
	// Number of thread blocks in grid 1024 height*width/blockSize
    gridSize = (int)ceil(data_size/(float)blockSize);
	
	// Execute the kernel add each thread for one output
    maxPerRow<<<gridSize, col_size>>>(d_a, d_c, data_size);
	
	// Copy result back to host
    cudaMemcpy( h_c, d_c, bytes_output, cudaMemcpyDeviceToHost );
	
	// print out data
	for(row_id = 0; row_id < col_size; row_id++){
		printf("%d ", (int)h_c[row_id]);
	}
	
	printf("\n done\n");
	
	// Release device memory
    cudaFree(d_a);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a); 
    free(h_c);
	
	
	return 0;
}