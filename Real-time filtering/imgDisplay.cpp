//
//  imgDisplay.cpp
//  CV
//
//  Created by Sumegha Singhania on 1/30/22.
//

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int, char**){
    string image_path = "/Users/sumeghasinghania/Desktop/CV/CV/Resources/try7.jpeg";
    Mat img,dst;
    img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        cout << "Could not access the image" << endl;
        return 1;
    }
    img.copyTo(dst);
    imshow("Original image", img);
    int k = waitKey(0);
    if(k == 'q')
    {
        imwrite("Image.jpeg", img);
    }
    
    return 0;

}
