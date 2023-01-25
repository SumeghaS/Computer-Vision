/*
 main.cpp
 Main function to execute core tasks: camera calibration and object projection
 and Extensions
 */
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "csv_util.h"
#include "Tasks.h"

using namespace cv;
using namespace std;

int main(int, char**){
    
    //flag to choose action
    int flag = 0;
    cout << "For calibration: 0" << endl;
    cout << "For projection: 1" << endl;
    cin >> flag;
    
    if(flag == 0){
        //calibrate camera
        calibrate_camera();
    }
    else{
        //virtual object insertion
        projection();
    }
    return 0;
}
