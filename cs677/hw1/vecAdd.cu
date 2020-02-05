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

// CUDA Kernel. Each thread adds a whole row.
__global__ void matAddRow(double *a, double *b, double *c, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
	// loop over your row
	int add = id*n;
	int i;
	for(i = 0; i < n; i++){
		c[add + i] = a[add + i] + b[add + i]; 
	}
}

// CUDA Kernel. Each thread adds a whole column.
__global__ void matAddCol(double *a, double *b, double *c, int n){
	// Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
	// loop over column
	int i;
	for( i = 0; i < n ; i++){
		c[i*n + id] = a[i*n + id] + b[i*n + id]; 
	}
}
 
int main( int argc, char* argv[] )
{
    // Size of matrices col * row
    int n = 1024;
	//how many data points there are 1024*1024
	int nsquared = 1048576;
 
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
    size_t bytes = nsquared*sizeof(double);
 
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
    for( row_id = 0; row_id < nsquared; row_id++ ) {
		h_a[row_id] = 4;
		h_b[row_id] = 1;
    }
	
	//print input matrix 
	int col_id;
	printf("INPUT A: \n");
	for(row_id = 0; row_id < 5; row_id++){
		for(col_id = 0; col_id < 5; col_id++){
			printf("%d ", (int)h_a[n*row_id + col_id]);
		}
		printf("\n");
	}
	
	printf("INPUT A: \n");
	
	for(row_id = 0; row_id < 5; row_id++){
		for(col_id = 0; col_id < 5; col_id++){
			printf("%d ", (int)h_b[n*row_id + col_id]);
		}
		printf("\n");
	}
 
    // Copy host matrices to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid 1024
    gridSize = (int)ceil((float)nsquared/blockSize);
 
    // Execute the kernel add each thread for one output
    matAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, nsquared);
 
    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
	//print out 5x5 
	printf("OUPUT C: \n");
	
    for( row_id = 0; row_id < 5; row_id++){
		for( col_id = 0; col_id < 5; col_id++){
			printf("%d ", (int)h_c[n*row_id + col_id]);
		}
		printf("\n");
	}
	
	// redo gridSize
	gridSize = (int)ceil((float)n/blockSize);
	
	// Execute the kernel, each thread for one row -- don't need to reset memory since dA and dB should be 1
	matAddRow<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
	
	// Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
	
	// execute other kernel for column -- same as above
	matAddCol<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
	
	// Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

    
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a); 
    free(h_b);
    free(h_c);
	
	printf("done\n");
 
    return 0;
}