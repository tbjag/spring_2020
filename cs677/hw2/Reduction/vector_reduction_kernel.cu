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

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define NUM_ELEMENTS 512


// **===----------------- MP3 - Modify this function ---------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, int n)
{
	__shared__ float partial_sum[NUM_ELEMENTS];
	
	//find id
	unsigned int t = threadIdx.x;
	
	//load from global into shared mem, do the first computation
	partial_sum[t] = g_data[t];
	
	for(unsigned int stride = 256; stride >= 1; stride >>= 1){
		__syncthreads();
		if(t < stride)
			partial_sum[t] += partial_sum[t+stride];
	}

	//put result into global
	if(t < 1)
		g_data[t] = partial_sum[t];
}

__global__ void reduction_adv(float *g_data, int n, int layer, int exp_2_less) //layer = 0 on first run
{
	//find id
	unsigned int t = threadIdx.x;
	
	//load from global into shared mem, do the first computation
	if(t < exp_2_less && n > t * 512^layer)
		g_data[t*512^layer] += g_data[t*512^layer + exp_2_less * 512^layer];
	
	for(unsigned int stride = exp_2_less/2; stride >= 1; stride >>= 1){ //n/2 doesnt work
		__syncthreads();
		if(t < stride)
			//partial_sum[t] += partial_sum[t+stride];
			g_data[t] += g_data[t%12^layer+stride*512^layer];
	}
}

__global__ void reduction_less(float *g_data, int n, int exp_2_less)
{
	//__shared__ float partial_sum[NUM_ELEMENTS/2];
	
	//find id
	unsigned int t = threadIdx.x;
	
	//load from global into shared mem, do the first computation
	if(t < exp_2_less && n > t + exp_2_less)
		g_data[t] += g_data[t + exp_2_less];
	
	for(unsigned int stride = exp_2_less/2; stride >= 1; stride >>= 1){ //n/2 doesnt work
		__syncthreads();
		if(t < stride)
			//partial_sum[t] += partial_sum[t+stride];
			g_data[t] += g_data[t+stride];
	}
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
