/******************************************************************************* 
 * Name        : SubstringDivisibility.java
 * Authors     : Theodore Jagodits, Alex Kubecka, Kurt Von Autenried
 * Version     : 1.0 
 * Date        : 1/29/2020
 * Description : parse permutations of pandigital strings
 * Link : https://projecteuler.net/problem=43
 * Pledge: "I pledge my honor that I have abided by the Stevens Honor System"
 * Reference: 
 * 
 * long start = System.nanoTime();
    // Your algorithm, including printing.
    System.out.printf("Elapsed time: %.6f ms\n", (System.nanoTime() - start) / 1e6);
 ******************************************************************************/
import java.util.*;
import java.io.*;
class SubstringDivisibility2 {

    static int divisors[] = new int[]{2,3,5,7,11,13,17};
    static long powers[] = new long[]{1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000};
    static long sum = 0;
    static long subNum;
    static int input[] = new int[11];
    static int numChars;
    static int temp;

    public static void permutations(int[] num, int length){
	int i,j;
	if(length == 1){
		if(panCheck(num) == 1) {
			for (j = 0 ; j < numChars ; j++) {
				sum += num[j]*powers[numChars-j-1];
			}
		}
	}
	else{
		if (length == 2) {
			swap(num,0,1);
			if(panCheck(num) == 1) {
				for (j = 0 ; j < numChars ; j++) {
					sum += num[j]*powers[numChars-j-1];
				}
			}
			swap(num,0,1);
			swap(num,1,1);
			if(panCheck(num) == 1) {
				for (j = 0 ; j < numChars ; j++) {
					sum += num[j]*powers[numChars-j-1];
				}
			}
		} else {
			for(i = 0; i < length ; i++){
				swap(num,i,length-1);
				permutations(num, length - 1);
				swap(num,i,length-1);
			}
		}
	}
    }

    private static void swap(int[] num, int swap1, int swap2){
	temp = num[swap1];
	num[swap1] = num[swap2];
	num[swap2] = temp;
    }

    public static int panCheck(int[] num){
	int numSize = numChars;
	if(numSize < 4)
		return 0;
	while(numSize >= 4){
		subNum = num[numSize-3]*100+num[numSize-2]*10+num[numSize-1];
		if(subNum%divisors[numSize-4] != 0){
			return 0;
		}
		numSize--;
	}
	return 1;
    }

    public static void main(String[] args){
	char[] inputc = args[0].toCharArray(); 
	numChars = args[0].length();
	for (int i = 0 ; i < numChars ; i++)
		input[i] = (int)(inputc[i]-48);
        //checks execution time
        long start = System.nanoTime();
        permutations(input, numChars);
        System.out.println("Sum: " + sum);
        System.out.printf("Elapsed time: %.6f ms\n", (System.nanoTime() - start) / 1e6);
    }
}