//
//  vidDisplay.cpp
//  CV
//
//  Created by Sumegha Singhania on 1/30/22.
//
#include <stdio.h>
#include "filter.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int, char**){
    
    //Video capture
    VideoCapture *capdev;

    // open the video device
    capdev = new VideoCapture(0);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    //Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
     //              (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_width = capdev->get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = capdev->get(cv::CAP_PROP_FRAME_HEIGHT);

    VideoWriter video("/Users/sumeghasinghania/Desktop/CV/CV/Screenshots/outcpp.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width,frame_height));
    //printf("Expected size: %d %d\n", refS.width, refS.height);

   // namedWindow("Video", 1); // identifies a window
    
    
    
    //Variables
    Mat frame,filter;
    Mat3s sob_x,sob_y,mag;
    int count=1;
    int flag=0;
    int levels=1;
    int threshold=255;
    string str="";
//---------------------------------------------------------------
    
    //Video display
    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        

        frame.copyTo(filter);
        //filters
        if(flag == 1){
            //grayscale
            cvtColor(frame, filter, COLOR_RGB2GRAY);
        }
        else if(flag == 2){
            //alt_grayscale
            alt_greyscale(frame, filter);
        }
        else if(flag == 3){
            //blur
            blur5x5(frame, filter);
        }
        else if(flag == 4){
            //sobelx
            frame.copyTo(sob_x);
            sobelX3x3(frame, sob_x);
            convertScaleAbs(sob_x, filter);
        }
        else if(flag == 5){
            //sobely
            frame.copyTo(sob_y);
            sobelY3x3(frame, sob_y);
            convertScaleAbs(sob_y, filter);
        }
        else if(flag == 6){
            //gradient magnitude
            frame.copyTo(sob_x);
            frame.copyTo(sob_y);
            frame.copyTo(mag);
            sobelX3x3(frame, sob_x);
            sobelY3x3(frame, sob_y);
            magnitude(sob_x, sob_y, mag);
            convertScaleAbs(mag, filter);
        }
        else if(flag == 7){
            //blur and quantize
            blurQuantize(frame, filter, levels);
        }
        else if(flag == 8){
            //cartoon
            cartoon(frame, filter, levels, threshold);
        }
        else if(flag == 9){
            //negative
            negative(frame, filter);
        }
        else if(flag == 10){
            flip(frame,filter);
        }
        else if(flag == 11){
            //split
            split(frame, filter);
        }
        else if(flag == 12){
            rbg_filter(frame, filter, str);
        }
        else if(flag == 13){
            sketch(frame, filter);
        }
        else if(flag == 14){
            color_pop(frame, filter);
        }
        
        video.write(filter);
        imshow("Video", filter);
        
      
//-----------------------------------------------------------------
        
        //keypress
        char key = cv::waitKey(10);
        
        //grayscale
        if(key == 'g'){
            flag=1;
        }
        
        //alternate grayscale
        if(key == 'h'){
            flag=2;
        }
        
        //Gaussian blur
        if(key == 'b'){
            flag=3;
        }
        
        //sobelx
        if(key == 'x'){
            flag=4;
        }
        
        //sobely
        if(key == 'y'){
            flag=5;
        }
        
        //gradient magnitude
        if(key == 'm'){
            flag=6;
        }
        
        //Blur and quantize
        if(key == 'l'){
            //take levels input from stdin
            cout << "Enter levels for quantization" << endl;
            cin >> levels;
            flag=7;
        }
        
        //cartoon
        if(key == 'c'){
            cout << "Enter levels for quantization" << endl;
            cin >> levels;
            cout << "Enter threshold" << endl;
            cin >> threshold;
            flag=8;
            
        }
        
        //negative
        if(key == 'n'){
            flag=9;
        }
        
        //flipped
        if(key == 'f'){
            flag=10;
        }
        
        //split
        if(key == 'p'){
            flag=11;
        }
    
        //revert
        if(key == 'r'){
            flag=0;
        }
        
        //rbg filter
        if(key == 'd'){
            cout << "Enter filter colour" << endl;
            cin >> str;
            flag=12;
        }
        
        //sketch
        if(key == 'k'){
            flag = 13;
        }
        
        //quit
        if( key == 'q') {
            break;
        }
        
        //colour_pop
        if(key=='e'){
            flag = 14;
        }
        
        
        //screenshot
        if (key == 's'){
            string str = "/Users/sumeghasinghania/Desktop/CV/CV/Screenshots/vid_image"+to_string(count)+".jpg";
            if(flag != 0){
                imwrite(str, filter);
            }
            else{
                imwrite(str, frame);
            }
            count++;
        }
    }
    video.release();
    delete capdev;
    return 0;
}
