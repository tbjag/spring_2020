import java.util.*;

public class Main{
    public static void main(String args[]){
        Scanner read = new Scanner(System.in);
        String numOfNumsS = read.next();//maybe type cast to make faster 
        String divisorS = read.next();
        int numOfNums = Integer.parseInt(numOfNumsS);
        int divisor = Integer.parseInt(divisorS);
        int count = 0;
        int[] values = new int[numOfNums];
        for(int i = 0; i<numOfNums; i++){
            String currNumS = read.next();
            int currNum = Integer.parseInt(currNumS);
            values[i] = currNum;
        }
        for(int j = 0; j < numOfNums; j++){
            if(values[j]%divisor == 0){
                count ++;
            }
        }
        read.close();
        System.out.println(count);
    }
}