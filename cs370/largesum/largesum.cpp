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
            //if end of line is not there \n, just sub 1
            int j = line.length() - 2;
            //cout<<line[j-1]<<'\n';
            // if(line.at(j-1) == '\n'){
            //     cout<<"test"; // not running this line
            //     j = j-2;
            // }else{
            //     j = j-2;
            // }
            cout << i << j << "\n";
            while(i >= 0 && j >= 0){
                digSum = sum[i] + line[j] - (2 * '0') + carry;
                carry = digSum/10;
                sum += digSum%10;
                i--;
                j--;
            }
            if(i > 0){
                for(i; i > 0; i--){
                    digSum = sum[i] - '0' + carry;
                    carry = digSum/10;
                    sum += digSum%10;
                }
            }else{
                for(j; j > 0; j--){
                    digSum = line[j] - '0' + carry;
                    carry = digSum/10;
                    sum += digSum%10;
                }
            }
        }

        myfile.close();
        cout << "sum is: " << sum << "\n";
        return 0;
    }
        
    else{
        cout << "Unable to open the file";
        return 1;
    }
    return 0;
}