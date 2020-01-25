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
//DOES NOT WORK ON WINDOES

int main(){
    //sum is reversed
    string sum = "0";
    string line;
    ifstream myfile ("test.txt");
    int carry = 0;
    int digSum = 0;
    //for testing
    int count = 0;

    if (myfile.is_open()){
        while ( getline (myfile,line) ){
            cout << "ITER: " << count << "\n";
            //reverse the line
            reverse(line.begin(), line.end());
            //find the lengths
            int sum_length = sum.length();
            // account for \n at end
            int line_length = line.length();
            int counter = 0;
            //continue if both are the same size
            while(counter < sum_length && counter < line_length){
                //2 * '0' = 96
                cout << "hello: " << line[counter] << " " << sum[counter] << "\n";
                digSum = (line[counter] - '0') + (sum[counter] - '0') + carry;
                carry = digSum/10;
                sum[counter] = digSum%10 + '0';
                //cout << "sum: " << digSum << counter << "\n";
                //inc counter
                counter++;
            }
            //where check if carry fits?
            //if line is bigger, add to sum
            while(counter < line_length){
                digSum = line[counter] + carry - '0';
                carry = digSum/10;
                sum += digSum%10 + '0';
                counter++;
            }
            //if sum is bigger add carry to it
            while(counter < sum_length){
                digSum = sum[counter] + carry - '0';
                carry = digSum/10;
                sum[counter] = digSum%10 + '0';
                counter++;
            }
            if(carry){
                sum += carry;
                carry = 0;
            }
            count++;
        }
        myfile.close();
        reverse(sum.begin(), sum.end());
        cout << "sum is: " << sum << "\n";
        return 0;
    }
    else{
        cout << "Unable to open the file";
        return 1;
    }
}