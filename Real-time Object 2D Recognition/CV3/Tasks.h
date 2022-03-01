//
//  Tasks.h
//  CV3
//
//  Created by Sumegha Singhania on 2/25/22.
//
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
#include <iterator>
#include <map>
#include "csv_util.h"

using namespace cv;
using namespace std;

#ifndef Tasks_h
#define Tasks_h


int thresholding(Mat &img, Mat &dst){
    
    //make saturated pixels dark
    Mat tmp,tmp2;
    img.copyTo(tmp2);
    cvtColor(img, tmp, COLOR_BGR2HSV);
    float m = 0.1;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(tmp.at<Vec3b>(i,j)[1]>95){ //if pixel is highly saturated
                for(int c=0;c<3;c++){
                    tmp2.at<Vec3b>(i,j)[c]=tmp2.at<Vec3b>(i,j)[c]*m;
                }
            }
        }
    }
    
    cvtColor(tmp2, dst, COLOR_BGR2GRAY);
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(dst.at<uchar>(i,j)>120){
                dst.at<uchar>(i,j)=0;
            }
            else{
                dst.at<uchar>(i,j)=255;
            }
        }
    }
    
    return 0;
}

int grassfire_growing(Mat &img, Mat &binary, int n){
    int dims[2] = {img.rows,img.cols};
    binary = Mat::zeros(2, dims, CV_16S);
    
    //pass1: traverse the thresholded image, if bg(0), assign min(left,up)+1, if fg(255) assign 0
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            //if bg pixel
            if(img.at<uchar>(i,j)==0){
                //edge cases
                if(i==0 && j==0){
                    //first pixel
                    binary.at<ushort>(i,j) = USHRT_MAX;
                }
                else if(i==0){
                    //first row
                    binary.at<ushort>(i,j)=USHRT_MAX;
                }
                else if(j==0){
                    //first column
                    binary.at<ushort>(i,j)=USHRT_MAX;
                }
                else{
                    int a = binary.at<ushort>(i-1,j); //top
                    int b = binary.at<ushort>(i,j-1); //left
                    if(a==b && a==USHRT_MAX){
                        binary.at<ushort>(i,j)=USHRT_MAX;
                    }
                    if(a<b){
                        binary.at<ushort>(i,j)=a+1;
                    }
                    else{
                        binary.at<ushort>(i,j)=b+1;
                    }
                }
                
            }
            
        }
    }
    
    //pass 2: traverse from bottom right, if bg(0), min(down,right)+1 if less than curr, add, if fg(255),0
    for(int i=img.rows-1;i>=0;i--){
        for(int j=img.cols-1;j>=0;j--){
            //if bg pixel
            if(img.at<uchar>(i,j)==0){
                //edge cases
                if(i==img.rows-1 && j == img.cols-1){
                    //do nothing
                }
                else if(i==img.rows-1){
                    //last row
                    int a = binary.at<ushort>(i,j+1)+1; //right
                    if(a<binary.at<ushort>(i,j)){
                        binary.at<ushort>(i,j)=a;
                    }
                }
                else if(j==img.cols-1){
                    //last col
                    int a = binary.at<ushort>(i+1,j)+1; //down
                    if(a<binary.at<ushort>(i,j)){
                        binary.at<ushort>(i,j)=a;
                    }
                }
                else{
                    int a = binary.at<ushort>(i+1,j)+1; //down
                    int b = binary.at<ushort>(i,j+1)+1; //right
                    if(a<b){
                        if(a<binary.at<ushort>(i,j)){
                            binary.at<ushort>(i,j)=a;
                        }
                    }
                    else{
                        if(b<binary.at<ushort>(i,j)){
                            binary.at<ushort>(i,j)=b;
                        }
                    }
                }
            }
            
        }
    }
    
    //processing on binary image
    //for n runs change all n val pixels to 255
    
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int run=1;
            while(run!=n){
                if(binary.at<ushort>(i,j)==run){
                    img.at<uchar>(i,j)=255;
                    break;
                }
                run++;
            }
            
        }
    }
    
    return 0;
}

int grassfire_shrinking(Mat &img, Mat &binary, int n){
    int dims[2] = {img.rows,img.cols};
    binary = Mat::zeros(2, dims, CV_16S);
    
    //pass1: traverse the thresholded image, if fg(255), assign min(left,up)+1, if bg(0) assign 0
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            //if fg pixel
            if(img.at<uchar>(i,j)==255){
                //edge cases
                if(i==0 && j==0){
                    //first pixel
                    binary.at<ushort>(i,j) = USHRT_MAX;
                }
                else if(i==0){
                    //first row
                    binary.at<ushort>(i,j)=USHRT_MAX;
                }
                else if(j==0){
                    //first column
                    binary.at<ushort>(i,j)=USHRT_MAX;
                }
                else{
                    int a = binary.at<ushort>(i-1,j); //top
                    int b = binary.at<ushort>(i,j-1); //left
                    if(a==b && a==USHRT_MAX){
                        binary.at<ushort>(i,j)=USHRT_MAX;
                    }
                    if(a<b){
                        binary.at<ushort>(i,j)=a+1;
                    }
                    else{
                        binary.at<ushort>(i,j)=b+1;
                    }
                }
                
            }
            
        }
    }
    
    //pass 2: traverse from bottom right, if bg(0), min(down,right)+1 if less than curr, add, if fg(255),0
    for(int i=img.rows-1;i>=0;i--){
        for(int j=img.cols-1;j>=0;j--){
            //if fg pixel
            if(img.at<uchar>(i,j)==255){
                //edge cases
                if(i==img.rows-1 && j == img.cols-1){
                    //do nothing
                }
                else if(i==img.rows-1){
                    //last row
                    int a = binary.at<ushort>(i,j+1)+1; //right
                    if(a<binary.at<ushort>(i,j)){
                        binary.at<ushort>(i,j)=a;
                    }
                }
                else if(j==img.cols-1){
                    //last col
                    int a = binary.at<ushort>(i+1,j)+1; //down
                    if(a<binary.at<ushort>(i,j)){
                        binary.at<ushort>(i,j)=a;
                    }
                }
                else{
                    int a = binary.at<ushort>(i+1,j)+1; //down
                    int b = binary.at<ushort>(i,j+1)+1; //right
                    if(a<b){
                        if(a<binary.at<ushort>(i,j)){
                            binary.at<ushort>(i,j)=a;
                        }
                    }
                    else{
                        if(b<binary.at<ushort>(i,j)){
                            binary.at<ushort>(i,j)=b;
                        }
                    }
                }
            }
            
        }
    }
    
    //processing on binary image
    //for n runs change all n val pixels to 0
    
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int run=1;
            while(run!=n){
                if(binary.at<ushort>(i,j)==run){
                    img.at<uchar>(i,j)=0;
                    break;
                }
                run++;
            }
            
        }
    }
    
    return 0;
}

int region_mapping(Mat &img, Mat &th, Mat &labels, Mat &stats, Mat &centroids){
    connectedComponentsWithStats(th, labels, stats, centroids);
    Mat connected_regions;
    img.copyTo(connected_regions);
    for(int i=0; i<stats.rows; i++)
    {
        int x = stats.at<int>(Point(0, i));
        int y = stats.at<int>(Point(1, i));
        int w = stats.at<int>(Point(2, i));
        int h = stats.at<int>(Point(3, i));
        int a = stats.at<int>(Point(4, i));
        double cx = centroids.at<double>(i, 0);
        double cy = centroids.at<double>(i, 1);
        
        Scalar color(rand()%255,rand()%256,rand()%256);
        Rect rect(x,y,w,h);
        rectangle(connected_regions, rect, color,LINE_8);
    }
//    imshow("regions", connected_regions);
    return 0;
}

//find contours of an image
//make oriented bounded boxes
int oriented_bb(Mat &img, Mat &binary, vector<vector<Point>> &contours){
    
    //contours of image
    findContours( binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    vector<RotatedRect> minRect( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
    }
    Mat drawing = Mat::zeros( binary.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
        //draw contours
        drawContours( drawing, contours, (int)i, color,LINE_8 );
        
        // draw min oriented bounding box
        Point2f rect_points[4];
        minRect[i].points( rect_points );
        for ( int j = 0; j < 4; j++ )
        {
            line( drawing, rect_points[j], rect_points[(j+1)%4], color,LINE_8 );
        }
    }
//    imshow( "All contours", drawing );
    return 0;
}

//1. run conditions to find major regions:
//contour size > n & ignore those with box along edges/or starting point
//2. find features
int major_regions(Mat &binary, vector<vector<Point>> &contours, vector<float> &img_data){
    vector<RotatedRect> minRect( contours.size() );
    Mat drawing = Mat::zeros( binary.size(), CV_8UC3 );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
    }
    for( size_t i = 0; i< contours.size(); i++ )
    {   int flag = 1;
        double area = minRect[i].size.height*minRect[i].size.width;
        if(area > 360000){
            Point2f rect_points[4];
            minRect[i].points( rect_points );
            for ( int j = 0; j < 4; j++ )
            {
                if(rect_points[j].x<10 || rect_points[j].y<10){
                    flag = 0;
                    break;
                }
            }
            if(flag == 1){
                //calculate features
                
                //1. oriented bb & ratio
                double bb_ratio=0;
                Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
                Scalar color2 = Scalar( rand()%255,rand()%255,rand()%255 );
                drawContours( drawing, contours, (int)i, color,LINE_8 );
                // rotated rectangle
                Point2f rect_points[4];
                minRect[i].points( rect_points );
                
                for ( int j = 0; j < 4; j++ )
                {
                    line( drawing, rect_points[j], rect_points[(j+1)%4], color,LINE_8 );
                }
                bb_ratio = minRect[i].size.width/minRect[i].size.height;

                
                //2,3 central axis and centroid
                Moments m = moments(contours[i]);
                double cx = m.m10/m.m00;
                double cy = m.m01/m.m00;
                double internal = (2*m.mu11)/(m.mu20-m.mu02);
                double thet = 0.5*atan(internal);
                //                thet=(thet/CV_PI)*180;
                
                //                double thet = (minRect[i].angle/180)*CV_PI;
                //                double cx = minRect[i].center.x;
                //                double cy = minRect[i].center.y;
                double cx2 = cx+1500*cos(thet);
                double cy2 = cy+1500*sin(thet);
                double cx3 = cx+1500*cos(thet+CV_PI);
                double cy3 = cy+1500*sin(thet+CV_PI);
                line(drawing, Point(cx3,cy3), Point(cx2,cy2), color2,2,LINE_8);
                
                
                //4. Hu moment invariants
                double huMoments[7];
                HuMoments(m, huMoments);
                //transform to comparable scale
                for(int i = 0; i < 7; i++) {
                    huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                
                }
                
                
                //5. percentage filled
                double perc = contourArea(contours[i])/area;
                    
                //all features of the region
                img_data.push_back(perc);
                img_data.push_back(bb_ratio);
//                img_data.push_back(cx);
//                img_data.push_back(cy);
                img_data.push_back(huMoments[0]);
                img_data.push_back(huMoments[1]);
                img_data.push_back(huMoments[2]);
                img_data.push_back(huMoments[3]);
                img_data.push_back(huMoments[4]);
                img_data.push_back(huMoments[5]);

            }
        }
    }
 
//    imshow( "Major regions", drawing );
 

    return 0;
}


#endif /* Tasks_h */
