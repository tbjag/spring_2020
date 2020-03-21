#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    //make vars
    int width = 1000;
    int height = 1000;
    float result[width][height], input1[height], input2[width][height], temp[width];
    clock_t start, end;
    float cpu_time_used;
    //random initialize for testing
    for(int o = 0; o < width; o++){
        input1[o] = 1;
        for(int p = 0; p < height; p++){
            input2[o][p] = 1;
        }
    }
	
	start = clock();
    //algorithm
    for(int i = 0; i<width; i++){
        temp[i] = 0.0f;
        for(int j =0; j < height; j++){
        	
            temp[i] += input2[i][j];
            result[i][j] = temp[i];
            //seperate threads
            for(int k = 0; k < height; k++){
                result[i][j] += input1[j] * input1[k];//speed up 
            }
        }
    }
    end = clock();
    
    
    //print output
    for(int i = 0; i < width; i++){
    	for(int j = 0; j < height; j++){
    		printf("%d ",(int)result[i][j]);
		}
		printf("\n");
	}
	printf("CPU time for execution: %lf ms\n", ((float)((end-start)*1000))/CLOCKS_PER_SEC);
    return 0;
}
