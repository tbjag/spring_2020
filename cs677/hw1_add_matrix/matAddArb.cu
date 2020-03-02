#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// CUDA kernel. Each thread takes care of one element of c
__global__ void matAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds - should be the same as before
    if (id < n)
        c[id] = a[id] + b[id];
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void matAddCol(double *a, double *b, double *c, int col_size, int row_size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds - should be the same as before
	// loop over column
	int i;
	for( i = 0; i < col_size; i++){
		c[i*row_size + id] = a[i*row_size + id] + b[i*row_size + id]; 
	}
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void matAddRow(double *a, double *b, double *c, int col_size, int row_size)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds - should be the same as before
	int add = id*col_size;
	int i;
    for(i = 0; i < row_size; i++){
		c[add+i] = a[add+i] + b[add+i]; 
	}
}

int main( int argc, char* argv[] ){
	
	// variable size of matrix < 1024
    int col_size = 10, row_size = 10;
	
	//check that sizes are correct
	if(col_size > 1025 || row_size > 1025){
		printf("size not valid\n");
		return 1;
	}
	
	//how many data points there are row*col
	int data_size = col_size * row_size;
 
    // Host input matrix
    double *h_a;
    double *h_b;
    //Host output matrix
    double *h_c;
 
    // Device input matrix
    double *d_a;
    double *d_b;
    //Device output matrix
    double *d_c;
 
    // Size, in bytes, of each matrix
    size_t bytes = data_size*sizeof(double);
 
    // Allocate memory for each matrix on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // Allocate memory for each matrix on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
	
	int row_id;
    // Initialize matrix on host
	// Simple initialize for now
    for( row_id = 0; row_id < data_size; row_id++ ) {
		h_a[row_id] = 4;
		h_b[row_id] = 2;
    }
	
	// Copy host matrices to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
	
	// Only thing that is different is to determine amount of warps/blocks/grid
	int blockSize, gridSize;
	
	// Number of threads in each thread block 
    blockSize = 1024;
	
	// Number of thread blocks in grid 1024 height*width/blockSize
    gridSize = (int)ceil(data_size/(float)blockSize);
	
	// Execute the kernel add each thread for one output
    matAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, data_size);
	
	// Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
	
	// Number of thread blocks, since 1 thread per row, dont need a larger grid
    gridSize = 1;
	
	// Execute the kernel add each thread for one output
    matAddCol<<<gridSize, col_size>>>(d_a, d_b, d_c, col_size, row_size);
	
	// Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
	
	// Execute the kernel add each thread for one output
    matAddRow<<<gridSize, row_size>>>(d_a, d_b, d_c, col_size, row_size);
	
	// Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
	
	int col_id;
    for( row_id = 0; row_id < 5; row_id++){
		for( col_id = 0; col_id < 5; col_id++){
			printf("%d ", (int)h_c[row_size*row_id + col_id]);
		}
		printf("\n");
	}
	
	// Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a); 
    free(h_b);
    free(h_c);
	
	return 0;
}