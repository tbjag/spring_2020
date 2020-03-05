#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"

#define DEFAULT_THRESHOLD  8000
#define TILE_SIZE 16

#define DEFAULT_FILENAME "BWstop-sign.ppm"


unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 
	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;
  
	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}

	// really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data. 
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);
	
	fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
	line += strlen(duh) + 1;

	long offset = line - chars;
	//lseek(fd, offset, SEEK_SET); // move to the correct offset
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	//long numread = read(fd, buf, bufsize);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

	fclose(fp);
	
	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++) 
		pic[i] = (int) buf[3*i];  // red channel
	
	return pic; // success
}




void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
	FILE *fp;
	int x,y;
  
	fp = fopen(filename, "wb");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
		exit(-1); 
	}
  
	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	fclose(fp);
}

__global__ void sobel_filter(int *inputM, int *outputM, int width, int height, int thresh){
	// shared 16x16
	__shared__ int local[TILE_SIZE][TILE_SIZE];
	
	//set up vars
	int tx = threadIdx.x; int ty = threadIdx.y;
	//find positions within input array for 14x14 offset by one for pixel border
	int row = blockIdx.x * 14 + tx + 1; 
	int col = blockIdx.y * 14 + ty + 1;
	
	//correct to the right place in input
	int place = (row-1)*width + col-1;

	//if in bounds of pic
	if((row - 1 < width) && (col - 1 < height)){
		//store into shared
		local[tx][ty] = inputM[place];
		//find inner part of 14x14
		if((tx > 0) && (tx < 15) && (ty > 0) && (ty < 15) && (row + 1 < width) && (col + 1 < height)){// <-- check reasoning
			//reuse vars
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
			
			// store into global -- hceck?
			outputM[place] = result;
		}
	}
}


int main( int argc, char **argv )
{
	int thresh = DEFAULT_THRESHOLD;
	char *filename;
	filename = strdup( DEFAULT_FILENAME);
  
	if (argc > 1) {
		if (argc == 3)  { // filename AND threshold
			filename = strdup( argv[1]);
			thresh = atoi( argv[2] );
		}
		if (argc == 2) { // default file but specified threshhold
			thresh = atoi( argv[1] );
		}
		fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
	}

	int xsize, ysize, maxval;
	unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval ); 
	
	int numbytes =  xsize * ysize * sizeof( int );//gets RID OF 3 *
	int *result = (int *) malloc( numbytes );
	if (!result) { 
		fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
		exit(-1); // fail
	}

	int i, j, magnitude, sum1, sum2; 
	
	for (int col=0; col<xsize; col++) {
		for (int row=0; row<ysize; row++) { 
			*result++ = 0; 
		}
	}

	for (i = 1;  i < ysize - 1; i++) {
		for (j = 1; j < xsize -1; j++) {
      
			int offset = i*xsize + j;

			sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
			+ 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
			+     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];
      
			sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
				- pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];
      
			magnitude =  sum1*sum1 + sum2*sum2;

			if (magnitude > thresh)
				result[offset] = 255;
			else 
				result[offset] = 0;
		}
	}

	write_ppm( "result8000gold.ppm", xsize, ysize, 255, result);
	fprintf(stderr, "sobel CPU done\n"); 

    // TO-DO: deallocate res and pic
	
	//Set up vars
	int *d_pic;
	int *d_res;
	int block_size;
	
	// Malloc on 
	cudaMalloc(&d_pic, numbytes);
	cudaMalloc(&d_res, numbytes);
	
	cudaMemcpy(d_pic, pic, numbytes, cudaMemcpyHostToDevice);
	
	block_size = 16;
	dim3 dim_block (block_size, block_size);
	//decrease dims by 2 to account for resizing
	dim3 dim_grid ((int)ceil((float)(xsize-2)/(block_size-2)), (int)ceil((float)(ysize-2)/(block_size-2)));
	
	// Run kernel 
	//sobel_filter<<< dim_grid, dim_block >>> (d_pic, d_res, xsize, ysize, thresh);
	
	// Copy result back to host
	cudaMemcpy(result, d_res, numbytes, cudaMemcpyDeviceToHost);
	
	// Free vars
	free(pic);
	free(result);
	//free(res);
	cudaFree(d_pic);
	cudaFree(d_res);
	
	// Write to output
	write_ppm( "result8000gpu.ppm", xsize, ysize, 255, result);
	fprintf(stderr, "sobel GPU done\n"); 
}




