#include <stdio.h>
int main(int argc, char** argv){
	if(argc > 3){
		printf("i am here\n");
	}
	int n = 5 / (argc - 1);
	return 0;

}



