#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main( int argc, char **argv ){
    //make vars
   	int row = atoi(argv[1]);
	int column = atoi(argv[2]);
    
    float result[row][column]; 
    float input1[column]; 
    float input2[row][column]; 
    float temp[row];
    clock_t start, end;
    float cpu_time_used;
    //random initialize for testing
    for (int o = 0; o < row; o++) {
        for(int p = 0; p < column; p++){
            input2[o][p] = 1.0f;
            result[o][p] = 0.0f;
        }
    }
    for(int i = 0; i<column; i++){
    	input1[i] = 1.0f;
	}
	
	start = clock();
    //algorithm
    for(int i = 0; i<row; i++){
        temp[i] = 0.0f;
        for(int j =0; j < column; j++){
        	
            temp[i] += input2[i][j];
            result[i][j] = temp[i];
            //seperate threads
            for(int k = 0; k < column; k++){
                result[i][j] += input1[j] * input1[k];//speed up 
            }
        }
    }
    end = clock();
    
    
    //print output
    for(int i = 0; i < row; i++){
    	for(int j = 0; j < column; j++){
    		printf("%d ",(int)result[i][j]);
		}
		printf("\n");
	}
	printf("CPU time for execution: %lf ms\n", ((float)((end-start)))/CLOCKS_PER_SEC);
    return 0;
}
