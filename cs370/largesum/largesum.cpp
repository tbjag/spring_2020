/******************************************************************************* 
 * Name        : largesum.cpp
 * Authors     : Theodore Jagodits, Alex Kubecka, Kurt Von Autenried
 * Version     : 1.0 
 * Date        : 1/21/2020
 * Description : add large 50 digit numbers together
 *  https://projecteuler.net/problem=13
 * "I pledge my honor that I have abided by the Stevens Honor System"
 ******************************************************************************/

#include <fstream>
#include <iostream>

using namespace std;

int main(){
    string line;
    ifstream myfile ("test.txt");
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
        cout << line << '\n';
        }
        myfile.close();
    } else{
        cout << "Unable to open the final";
    }
    return 0;
}