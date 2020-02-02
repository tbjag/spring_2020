/******************************************************************************* 
 * Name        : SubstringDivisibility.java
 * Authors     : Theodore Jagodits, Alex Kubecka, Kurt Von Autenried
 * Version     : 1.0 
 * Date        : 1/29/2020
 * Description : parse permutations of pandigital strings
 * Link : https://projecteuler.net/problem=43
 * Pledge: "I pledge my honor that I have abided by the Stevens Honor System"
 * Reference: https://www.geeksforgeeks.org/johnson-trotter-algorithm/
 ******************************************************************************/
import java.lang.*;

class SubstringDivisibility{
    public static int divisors[] = new int[]{2,3,5,7,11,13,17};
    public static long sum = 0;

    //declared booleans to show direction -- might remove -- just for readability
    private final static boolean LEFT_TO_RIGHT = true; 
    private final static boolean RIGHT_TO_LEFT = false;
    

    //finds the factorial so that we don't need to calculate more perms
    public static int factorial(int n){
        int i = 1;
        int sum = 1;
        for( ; i < n; i++){
            sum *= i;
        }
        return sum;
    }

    //checks for the substring modifications
    public static boolean panCheck(int[] num){
        

        return true;
    }

    //find the largest mobile int
    public static int findLargestMobileIdx(int[] num, boolean[] dir, int n){
        int i;
        //forloop bounded 1 to n-1 instead of 0 to n
        for(i = 1; i < n-1; i++){
            //look left?
            if(){

            }
        }


        return 0;
    }
    
    //finds the permutations
    public static void onePermutation(int[] num, boolean[] direction, int n){
        //
    }


    public static void main(String[] args){
        //checks execution time
        long start = System.nanoTime();
        //checks if correct amount of args
        if(args.length != 1){
            System.out.printf("incorrect amount of args\n");
            return;
        }
        //check int size
        int numSize = args[0].length();
        if(numSize <= 3 || numSize > 10){
            System.out.printf("Input must be at least 4 digits, at most 10\n");
            return;
        } 
        //initialize array to read in to function
        int[] buf = new int[numSize];
        int i;
        for(i = 0 ; i < numSize; i++){
            //read in each value into buffer <--might be problem
            buf[i] = args[0].charAt(i);
        }
        //initialize bool array facing right to left
        boolean[] direction = new boolean[numSize];
        for(i = 0 ; i < numSize; i++){
            direction[i] = RIGHT_TO_LEFT;
        }
        //generate permutations and build sum
        for(i = 0; i < factorial(numSize); i++){
            onePermutation(buf, direction, numSize);
        }

        System.out.println("Sum: " + sum);
        System.out.printf("Elapsed time: %.6f ms\n", (System.nanoTime() - start) / 1e6);
    }
    
}
