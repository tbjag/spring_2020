/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* This software and the information contained herein is PROPRIETARY and 
* CONFIDENTIAL to NVIDIA and is being provided under the terms and 
* conditions of a Non-Disclosure Agreement.  Any reproduction or 
* disclosure to any third party without the express written consent of 
* NVIDIA is prohibited.     
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

#ifdef _WIN32
#define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// int readFile(filename:str, data: array);
#include "file_io.h"

// includes, kernels
#include <vector_reduction_kernel.cu>
//#include <vector_reduction_kernel_adv.cu>

// For simplicity, just to get the idea in this MP, we're fixing the problem size to 512 elements.
#define NUM_ELEMENTS 512
#define BLOCK_SIZE 32

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

float computeOnDevice(float* h_data, int array_mem_size);

extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        
        case 1:  // One Argument
            errorM = readFile(argv[1], h_data);
            if(errorM != 1)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    float reference = 0.0f;  
    computeGold(&reference , h_data, num_elements);
    
    // **===-------- Modify the body of this function -----------===**
    float result = computeOnDevice(h_data, num_elements);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    float epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, executes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice(float* h_data, int num_elements)
{
	// num is zero or 1 just return solution
	if(num_elements == 0)
		return 0.0f;
	else if (num_elements == 1)
		return h_data[0];
	else if (num_elements % 2 != 0)
		num_elements += 1; //will just have a 0 at the end
	
	// declare device vector 
	float *d_data;
	
	// calc number of bytes
	size_t bytes = num_elements * sizeof(float);
	
	// malloc on device
	cudaMalloc(&d_data, bytes);
	
	// copy data to device
	cudaMemcpy( d_data, h_data, bytes, cudaMemcpyHostToDevice);
	
	int block_size, grid_size, half_elements;
	
	//calc block size and grid size 
	//block_size = 256;
	//grid_size = (int)ceil((num_elements/2)/(float)block_size);
	//block_size = num_elements/2;
	//grid_size = (int)ceil();
	half_elements = num_elements/2;
	
	printf("%d\n", block_size);
	//send to appropriate function 
	if(num_elements <= 512){
		//appropriate block size 
		block_size = (half_elements) % 32 == 0 ? half_elements : half_elements + (BLOCK_SIZE - half_elements%BLOCK_SIZE); 
		reduction<<<1, block_size >>>(d_data, num_elements);
	} else{
		//work on this
		block_size = 256;
		//grid_size = (int)ceil((num_elements/2)/(float)block_size);
		//reduction_adv<<<grid_size, block_size >>>(d_data, num_elements);
	}

	// Copy result back to host
	cudaMemcpy( h_data, d_data, bytes, cudaMemcpyDeviceToHost );
	
	// print out result
	for(int i = 0; i < NUM_ELEMENTS; i++){
		printf("%lf ", h_data[i]);
	}
	printf("\n");
	
	// release memory
	cudaFree(d_data);
	
	// return single point
	return h_data[0];

}
     
