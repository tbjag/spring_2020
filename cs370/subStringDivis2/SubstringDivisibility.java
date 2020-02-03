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
        int res = 1;
        for( ; i <= n; i++){
            res *= i;
        }
        return res;
    }

    //checks for the substring modifications
    public static void panCheck(int[] num, int n){
        boolean check = true;
        int mult = 1000, i = 4;
        int total = num[0] + num[1]*10 + num[2]*100 + num[3] * 1000;
        for(; i < ){

        }
    }

    //find the largest mobile int
    public static int findLargestMobileIdx(int[] arr, boolean[] dir, int n){
        int i;
        int mobile = -1;
        int mobilePos = 0;
        //maybe bound for loop???? -- check later
        for(i = 0; i < n; i++){
            //look in direction of bool arr and check size
            if(i != 0 && dir[i] == RIGHT_TO_LEFT ){
                if(arr[i-1] < arr[i] && mobile < arr[i]){
                    mobile = arr[i];
                    mobilePos = i;
                }  
                //look the other direction
            } else if(i != n-1 && dir[i] == LEFT_TO_RIGHT){
                if(arr[i+1] < arr[i] && mobile < arr[i]){
                    mobile = arr[i];
                    mobilePos = i;
                } 
            }
        }
        return mobilePos;
    }
    
    //finds the permutations
    public static void onePermutation(int[] arr, boolean[] direction, int n){
        //find the largest mobile int pos
        int mobilePos = findLargestMobileIdx(arr, direction, n);
        int mobileVal = arr[mobilePos];
        int temp;
        boolean tempbool;
        //pointing to left
        if(direction[mobilePos] == RIGHT_TO_LEFT){
            //swap
            temp = arr[mobilePos];
            arr[mobilePos] = arr[mobilePos-1];
            arr[mobilePos-1] = temp;
            //swap bools 
            tempbool = direction[mobilePos];
            direction[mobilePos] = direction[mobilePos-1];
            direction[mobilePos-1] = tempbool;
        //pointing to the right
        }else if(direction[mobilePos] == LEFT_TO_RIGHT){
            temp = arr[mobilePos];
            arr[mobilePos] = arr[mobilePos+1];
            arr[mobilePos+1] = temp;
            //swap bools
            tempbool = direction[mobilePos];
            direction[mobilePos] = direction[mobilePos+1];
            direction[mobilePos+1] = tempbool;
        }
        //swap dir elements greater than mobile PROBLEM
        for(int i = 0; i < n; i++){
            if(arr[i] > mobileVal){
                if(direction[i] == LEFT_TO_RIGHT){
                    direction[i] = RIGHT_TO_LEFT;
                }else if(direction[i] == RIGHT_TO_LEFT){
                    direction[i] = LEFT_TO_RIGHT;
                }
            }
        }
        //permutation done...check if hits params
        panCheck(arr, n);
    }

    public static void main(String[] args){
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
            //read in each value into buffer
            buf[i] = args[0].charAt(i) - 48;
        }
        //initialize bool array facing right to left
        boolean[] direction = new boolean[numSize];
        for(i = 0 ; i < numSize; i++){
            direction[i] = RIGHT_TO_LEFT;
        }
        //checks execution time
        long start = System.nanoTime();
        //check the first permutation and add to sum if it hits reqs
        panCheck(buf, numSize);
        //generate permutations and build sum
        for(i = 1; i < factorial(numSize); i++){
            onePermutation(buf, direction, numSize);
        }
        //print sum
        System.out.println("Sum: " + sum);
        System.out.printf("Elapsed time: %.6f ms\n", (System.nanoTime() - start) / 1e6);
    }  
}
