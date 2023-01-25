/*
 Harris_corners.cpp
 Contains functions corresponding to finding Harris corners in live video
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

//function to detect harris corners and draw them on the frame
int detectHarris(int thresh, Mat &gray, Mat &frame){
   
    int blockSize = 2;
    int aperture = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(gray.size(), CV_32FC1);
    
    //detect corners
    cornerHarris(gray, dst, blockSize, aperture, k);
    
    //normalize the destination image and convert scale
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    
    //draw circles corresponding to corners on frame
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                cv::circle( frame, cv::Point(j,i), 5,  cv::Scalar(255), 2, 8, 0 );
            }
        }
    }
    return 0;
}

int main(int, char**){
    
    //counter to save images
    int count = 1;
    
    Mat frame;
    int thresh = 200;

    //Video capture
    VideoCapture *capdev;
    
    // open the video device
    capdev = new VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    namedWindow("Video", 1);
    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        //call detection function
        detectHarris(thresh,gray,frame);
        imshow("Video", frame);
        char key = cv::waitKey(10);
        
        //quit of keypress 'q'
        if( key == 'q') {
            break;
        }
        
        //save frame if ketpress 's'
        if(key == 's'){
            string str = "/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/Calib_frames/harrison"+to_string(count)+".jpg";
            imwrite(str, frame);
            count++;
        }
        
    }
    
    delete capdev;
    return 0;
}
