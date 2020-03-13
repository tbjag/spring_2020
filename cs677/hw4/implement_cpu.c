#include <stdio.h>
#include <stdlib.h>

int print_output(float output[128][128]){
    //print output
    for(int o = 0; o < 128; o++){
        for(int p = 0; p < 128; p++){
           printf("%f ", output[o][p]);
        }
        printf("\n");
    }
}

int main(){
    //make vars
    float result[128][128], input1[128], input2[128][128], temp[128];
    //random initialize for testing
    for(int o = 0; o < 128; o++){
        input1[o] = 1;
        for(int p = 0; p < 128; p++){
            input2[o][p] = 2;
        }
    }

    //algorithm
    for(int i = 0; i<128; i++){
        temp[i] = 0.0;
        for(int j =0; j < 128; j++){
            temp[i] += input2[i][j];
            result[i][j] = temp[i];
            for(int k = 0; k < 128; k++){
                result[i][j] += input1[j] * input1[k];//what
            }
        }
    }

    //print output
    print_output(result);
    return 0;
}
