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
    int carry = 0;
    int digSum = 0;

    if (myfile.is_open()){
        while ( getline (myfile,line) ){
            int i = sum.length() - 1;
            //if end of line is not there \n, just sub 1 ISSUE: when no new line
            int j = line.length() - 2;
            cout << "i,j : " << i << j << "\n";
            while(i >= 0 && j >= 0){
                digSum = sum[i] + line[j] - (2 * '0') + carry;
                carry = digSum/10;
                sum[i] = digSum%10 + '0';
                i--;
                j--;
            }
            //add carry here if no more chars in sum
            cout << i << "\n";
            if(i < 0 && carry > 0){
                cout << "here" << "\n";
                sum += carry + '0';
                carry = 0;
            }
            //if digits left over for sum then add
            while(i >= 0){
                digSum = sum[i] - '0' + carry;
                carry = digSum/10;
                sum[i] = digSum%10 + '0';
                i--;
            }
            //if digits left over in line then add to sum
            while(j >= 0){
                digSum = line[j] - '0' + carry;
                carry = digSum/10;
                sum += digSum%10 + '0';
                j--;
            }
        }
        myfile.close();
        cout << carry << "\n";
        if(carry){
            sum += carry + '0';
        }
        reverse(sum.begin(), sum.end());
        cout << "sum is: " << sum << "\n";
        return 0;
    }
        
    else{
        cout << "Unable to open the file";
        return 1;
    }
}