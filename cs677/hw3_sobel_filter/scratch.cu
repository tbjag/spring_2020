// Check within bounds
	if(idx_y + 1< height && idx_x + 1< width && idx_y - 1 >= 0 && idx_x -1 >= 0){
		// find gx within bounds
		//base case left
		if(tx == 0){
			// if bottom left
			if(ty + 1 == blockDim.y){
				gx = 
				gy = 
			}else if(ty == 0){
				int top_left = inputM[(idx_x-1)*width + idx_y -1];
				int bot_left = inputM[(idx_x+1)*width + idx_y -1];
				int top_right = inputM[(idx_x-1)*width + idx_y + 1];
				//here gx
				gx = top_left - top_right + (2*inputM[idx_x*width + idx_y -1]) - (2*local[tx+1][ty]) + bot_left - local[tx+1][ty+1];
				gy = 
			}else{
				// verified
				int top_left = inputM[(idx_x-1)*width + idx_y -1];
				int bot_left = inputM[(idx_x+1)*width + idx_y -1];
				gx = top_left - local[tx+1][ty-1] + (2*inputM[idx_x*width + idx_y -1]) - (2*local[tx+1][ty]) + bot_left - local[tx+1][ty+1];
				gy = top_left + (2 * local[tx][ty-1]) + local[tx+1][ty-1] - bot_left - 2*local[tx][ty+1] - local[tx+1][ty+1];
			}	
		}else if(tx +1 == blockDim.x){ 
			
		}else{
			
		}
	}
	
	magnitude = gx*gx + gy*gy;
	
	if(magnitude > thresh)
		result = 255;
	else
		result = 0;
		
local[tx][ty] = inputM[place];
		
		int gx = 0,gy = 0;
		int magnitude, result;
	
		//check bounds within 1 px border -- USE PLACE
	
		if((tx == 0 && ty == 0)||(tx==0&&ty+1<blockDim.y)||(tx+1<blockDim.x && ty == 0)||(tx+1<blockDim.x && ty+1<blockDim.y)){
			//top left global load
			
		}else if(tx == 0){
			
		}else if(ty == 0){
			
		}else if(tx + 1 == blockDim.x){
			
		}else if(ty + 1 == blockDim.y){
			
		}else{
			//no edges or corners
			int top_left = local[tx-1][ty-1], top_right = local[tx+1][ty-1], bot_left = local[tx-1][ty+1], bot_right = local[tx+1][ty+1];
			gx = top_left - top_right + 2*local[tx-1][ty] - 2*local[tx+1][ty] + bot_left - bot_right;
			gy = top_left + 2*local[tx][ty-1] + top_right - bot_left - 2*local[tx][ty+1] - bot_right;
		}
		
---------------
// Copy data to device
	

	block_size = 16;
	dim3 dim_block (block_size, block_size);
	//decrease dims by 2 to account for resizing
	dim3 dim_grid ((int)ceil((float)(xsize-2)/(block_size-2)), (int)ceil((float)(ysize-2)/(block_size-2)));
	printf("going intp func\n");
	// Run kernel 
	//sobel_filter<<< dim_grid, dim_block >>> (d_pic, d_res, xsize, ysize, thresh);
	printf("going out of func\n");