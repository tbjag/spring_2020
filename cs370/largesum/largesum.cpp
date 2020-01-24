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
    string sum = "";
    string line;
    ifstream myfile ("test.txt");
    //Theo way
    /*if (myfile.is_open())
    {
        while ( getline (myfile,line) ){
        //reverse the line
        reverse(line);
        //in every line
        for(int i = 0; i < line.length(); i++){
            //string t = line[i] + 
            }
        }
        myfile.close();
    }*/

    //Kurt way
    char curr;
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
    myfile.close();
    return sum;
    else{
        cout << "Unable to open the file";
    }
    return 0;
}