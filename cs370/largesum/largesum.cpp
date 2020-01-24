/******************************************************************************* 
 * Name        : largesum.cpp
 * Authors     : Theodore Jagodits, Alex Kubecka, Kurt Von Autenried
 * Version     : 1.0 
 * Date        : 1/21/2020
 * Description : add large 50 digit numbers together
 *  https://projecteuler.net/problem=13
 * "I pledge my honor that I have abided by the Stevens Honor System"
 * reference: https://www.geeksforgeeks.org/sum-two-large-numbers/
 ******************************************************************************/

#include <fstream>
#include <iostream>
#include <bits/stdc++.h>
#include <string>//maybe look for \0 instead of this

using namespace std;

//reverse string function??


int main(){
    string sum = "0";
    string line;
    ifstream myfile ("test.txt");
    //Theo way
    int carry = 0;
    int digSum = 0;
    if (myfile.is_open())
    {
        while ( getline (myfile,line) ){
        //reverse the line
            reverse(line);
            //in every line
            int i = sum.length() - 1;
            int j = line.length() - 1;
            while(i >= 0 && j >=0){
                digSum = ((sum[i] - '0') + (line[j] - '0')+ carry);
                sum +=(digSum %10+ '0');
                carry = digSum/10;
                i--;
                j--;
            }
            //if num2 is smaller add remaining digit of num1 to res
           while(i >=0){
            digSum =((sum[i] - '0') + carry);
            sum +=(digSum%10 + '0');
            carry = digSum/10;
            i--;
           }
            //if num1 is smaller add remaining digit of num2 to res
           while(j >=0){
            digSum = ((line[j] - '0') + carry);
            sum += (digSum%10 + '0');
            carry = digSum/10;
            j--;
           }
            //at last if carry is there add it to res
           if(carry){
            sum +=(carry + '0');
           }
            //finally reverse res string to get the final sum
            reverse(sum);
        }
    }
    myfile.close();
    

    //Kurt way
    /*char curr;
    int carry = 0;
    int digSum = 0;
    if(myfile.is_open()){
        while(myfile.get(curr)){
            if(curr != '/n'){
                int i = sum.length() - 1;
                digSum = ((sum[i] - '0') + (curr - '0'));
                sum[i] = (digSum % 10 + '0');
                carry = digSum/10;
                while(carry){
                    i--;
                    digSum = (sum[i] - '0') + carry;
                    sum[i] = (digSum % 10 + '0');
                    carry = digSum/10;
                }
            }
        }
    }
    myfile.close();*/
    return sum;
    else{
        cout << "Unable to open the file";
    }
    return 0;
}