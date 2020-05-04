/* 
    Name          : biker_racer.cpp
    Authors       : Theodore Jagodits, Alex Kubecka, Kurt Von Autenried
    Date          : 5/1/2020
    Description   : There are N bikers present in a city (shaped as a grid) having M bikes. 
                    All the bikers want to participate in the HackerRace competition, but 
                    unfortunately only K bikers can be accommodated in the race. Jack is 
                    organizing the HackerRace and wants to start the race as soon as possible. 
                    He can instruct any biker to move towards any bike in the city. 
                    In order to minimize the time to start the race, Jack instructs the bikers 
                    in such a way that the first K bikes are acquired in the minimum time.
    Link :        : https://www.hackerrank.com/challenges/bike-racers/problem
    Pledge:       : "I pledge my honor that I have abided by the Stevens Honor System"
 */

#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>
using namespace std;

//create node that holds data
class bike_node{
    //access specifier
    public:
    // data members
    int biker, bike, distance;
    //replace function to compare??
    bool operator<(const bike_node& r)const{
        return distance<r.distance;
    }
};


bool depth_first_search(int u, vector<bool> used, vector<vector<int>> g, vector<int> lines, int num_bikes){
    for(int i = 0; i < num_bikes; i++){
        if(g[u][i] && !used[i]){
            used[i] = true;
            if(lines[i] == -1 || depth_first_search(lines[i], used, g, lines, num_bikes)){
                lines[i] = u;
                return true;
            }
        }
    }
    return false;
}


//munkers algorithm iterative
int munkers(int midpoint, vector<bike_node> agg_nodes, int num_bikers, int num_bikes){
    //create vector g? explain
    vector<vector<int>> g;
    g.resize(510, vector<int>(510, 0));
    //fill vector with spots
    for(int i = 0; i <= midpoint; i++){
        g[agg_nodes[i].biker][agg_nodes[i].bike] = 1;
    }

    int result = 0;
    
    vector<int> lines;
    vector<bool> used;
    
    lines.resize(num_bikes, -1);
    used.resize(num_bikes);

    

    //munkers...
    for(int i = 0; i < num_bikers; i++){
        fill(used.begin(), used.end(), false);
        if(depth_first_search(i, used, g, lines, num_bikes)){
            result++;
        }
    }
    return result;
}


//set up and solve through binary search
int binary_search(vector<tuple<int, int>> biker_vector, vector<tuple<int, int>> bikes_vector, int allowed){
    int num_bikers = biker_vector.size();
    int num_bikes = bikes_vector.size();

    //fill vector with all distances from each bike
    vector<bike_node> all_nodes;
    for(int i = 0; i < num_bikers; i++){
        for(int j = 0; j < num_bikes; j++){
            bike_node temp;
            temp.biker = i;
            temp.bike = j;
            //maybe speedup
            temp.distance = 
            (get<0>(biker_vector[i])-get<0>(bikes_vector[j]))
            *(get<0>(biker_vector[i])-get<0>(bikes_vector[j]))
            +(get<1>(biker_vector[i])-get<1>(bikes_vector[j]))
            *(get<1>(biker_vector[i])-get<1>(bikes_vector[j]));                                       
            all_nodes.push_back(temp);
        }
    }

    //sort based on distance
    sort(all_nodes.begin(), all_nodes.end());

    // initialize 
    int low = -1, high = all_nodes.size()-1, middle;

    while(high - low > 1){
        middle = (high + low)/2;
        cout << middle << " " << high << " " << low << endl;
        if(calc(middle, all_nodes, num_bikers, num_bikes) >= allowed){
            high = middle;
        }else{
            low = middle;
        }
    }

    return all_nodes[high].distance;
}

// read into program
int main(){
    //declare vars
    int num_bikers, num_bikes, bikes_allowed, count, final_answer;
    vector<tuple<int, int>> biker_vector, bikes_vector;

    //speedup
    ios::sync_with_stdio(0);
    cin.tie(NULL);

    //input 
    cin >> num_bikers >> num_bikes >> bikes_allowed;

    //fill vectors
    int temp1, temp2;
    for(count = 0; count < num_bikers; count++){
        cin >> temp1 >> temp2;
        biker_vector.push_back(make_tuple(temp1,temp2));
    }
    for(count = 0; count < num_bikes; count++){
        cin >> temp1 >> temp2;
        bikes_vector.push_back(make_tuple(temp1,temp2));
    }
    
    //call solution
    final_answer = binary_search(biker_vector, bikes_vector, bikes_allowed);
    cout << final_answer << endl;
    return 0;

}