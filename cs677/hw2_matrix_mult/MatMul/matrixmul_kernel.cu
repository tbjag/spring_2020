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

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

//change tile width
#define TILE_WIDTH 16

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P, int num_tiles, int n_size, int m_size)
{
	// create shared memory space to store used vals
	__shared__ float m_local_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ float n_local_shared[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x; int by = blockIdx.y; 
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	//identifies which tile in section you are working in -- correlates to P
	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;
	
	float temp_val = 0;
	
	for(unsigned int count = 0; count < num_tiles; count++){
		unsigned int m_find = row*M.width + (count*TILE_WIDTH + tx);
		unsigned int n_find = (count*TILE_WIDTH + ty)*N.width + col;
		if(m_find < m_size && n_find < n_size){ // check if in bounds
			m_local_shared[ty][tx] = M.elements[m_find]; //find the right section for m!
			n_local_shared[ty][tx] = N.elements[n_find]; //find the right section for n!
			
			__syncthreads();
			
			for(int k =0; k < TILE_WIDTH; k++)
				temp_val += m_local_shared[ty][k] * n_local_shared[k][tx];
			
			__syncthreads();
		}
	}
	
	P.elements[row*M.width + col] = temp_val;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
