#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "tasks.hpp"

using namespace cv;
using namespace std;

int main(int, char**){
    
    int task_flag=0;
    string target_path;
    Mat target,tmp,dst;
    vector<sort_data_type> dist;
    vector<sort_data_type> dist2;
    vector<sort_data_type> dist3;
    vector<sort_data_type> dist4;
    vector<sort_data_type> dist5;

    int ext_flag = 0;
    cout << "Enter target image path" << endl;
    cin >> target_path;
    target = imread(target_path, IMREAD_COLOR);
    
    cout << "Enter task number" << endl;
    cin >> task_flag;
    
    if(task_flag == 1){
        vector<sort_data_type> dist;
        Task1(target, &dist);
    }
    else if(task_flag == 2){
        vector<sort_data_type> dist;
        Task2(target, &dist);
    }
    else if(task_flag == 3){
        vector<sort_data_type> dist;
        vector<sort_data_type> dist2;
        Task3(target, &dist, &dist2);
    }
    else if(task_flag == 4){
        vector<sort_data_type> dist;
        vector<sort_data_type> dist2;
        Task4(target, &dist, &dist2);
    }
    else if(task_flag == 5){
        vector<sort_data_type> dist;
        vector<sort_data_type> dist2;
        vector<sort_data_type> dist3;
        vector<sort_data_type> dist4;
        vector<sort_data_type> dist5;
        square301x301(target, tmp);
        Task5(tmp, &dist, &dist2, &dist3, &dist4, &dist5);
    }
    else{
        cout << "Enter the following for extensions" << endl;
        cout << "6: Gabor filter" << endl;
        cout << "7: Laws filter" << endl;
        cin >> ext_flag;

        if(ext_flag == 6){
            vector<sort_data_type> dist;
            vector<sort_data_type> dist2;
            gab_texture(target, &dist, &dist2);
        }
        else if(ext_flag == 7){
            vector<sort_data_type> dist;
            vector<sort_data_type> dist2;
            vector<sort_data_type> dist3;
            vector<sort_data_type> dist4;
            vector<sort_data_type> dist5;
            laws_fil(target, &dist, &dist2, &dist3, &dist4, &dist5);
        }

    }
    
    return 0;

}
