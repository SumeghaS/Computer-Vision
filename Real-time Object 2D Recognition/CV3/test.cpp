////
////  test.cpp
////  CV3
////
////  Created by Sumegha Singhania on 2/26/22.
////
//
//#include <stdio.h>
//#include <cstdio>
//#include <cstring>
//#include <cstdlib>
//#include <dirent.h>
//#include <stdio.h>
//#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//#include "Tasks.h"
//
//using namespace cv;
//using namespace std;
//
//int main(int, char**) {
//   Mat img = imread("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/Resources/final/test1.jpg", IMREAD_COLOR);
//    Mat tmp,binary;
//    thresholding(img, tmp);
//    imshow("binary image",tmp);
//    
//    //clean up binary image
//    grassfire_shrinking(tmp, binary, 2);
//    grassfire_growing(tmp, binary,30);
//    grassfire_shrinking(tmp, binary, 20);
//    imshow("after cleanup",tmp);
//    
//    //segmentation
//    Mat labels;
//    Mat stats;
//    Mat centroids;
//    region_mapping(img, tmp, labels, stats, centroids);
//    imshow("regions",img);
//    
//    //cropping major area
//    int x = stats.at<int>(Point(0, 1));
//    int y = stats.at<int>(Point(1, 1));
//    int w = stats.at<int>(Point(2, 1));
//    int h = stats.at<int>(Point(3, 1));
//    double cx = centroids.at<double>(1, 0);
//    double cy = centroids.at<double>(1, 1);
//    Mat major_region = tmp(Rect(Point(x, y), Point(x+w, y+h)));
//    imshow("cropped major area", major_region);
//    
//    
//    //finding contours only for the major region
//    vector<vector<Point>> contours;
//    findContours( major_region, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
//    //find largest contour
//    double max=0;
//    int max_i=0;
//    for(int i=0;i<contours.size();i++){
//        if(contours[i].size()>max){
//            max = contours[i].size();
//            max_i=i;
//        }
//    }
//    
//    
//    
//    //oriented bounding box
//    RotatedRect oriented_bb = minAreaRect(contours[max_i]);
//    //change center to actual centroid
//    oriented_bb.center = Point(cx,cy);
//    Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
//    Point2f rect_points[4];
//    oriented_bb.points( rect_points );
//    for ( int j = 0; j < 4; j++ )
//    {
//        line( img, rect_points[j], rect_points[(j+1)%4], color,LINE_8 );
//    }
//    
//    //central axis
////    Moments mu = moments( contours[max_i] );
////    double internal = (2*mu.mu11)/(mu.mu20-mu.mu02);
////    double thet = 0.5*atan(internal);
//    double theta=(oriented_bb.angle/180)*CV_PI;
//    double cx2 = cx+1500*cos(theta);
//    double cy2 = cy+1500*sin(theta);
//    double cx3 = cx+1500*cos(theta+CV_PI);
//    double cy3 = cy+1500*sin(theta+CV_PI);
//    line(img, Point(cx3,cy3), Point(cx2,cy2), Scalar(255,0,0),2,LINE_8);
//    imshow("axis", img);
//
//    
////    vector<RotatedRect> minRect( contours.size() );
////    for( size_t i = 0; i < contours.size(); i++ )
////        {
////            minRect[i] = minAreaRect( contours[i] );
////        }
//
//    
////    for( size_t i = 0; i< contours.size(); i++ )
////        {
////            Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
////            // rotated rectangle
////            Point2f rect_points[4];
////            minRect[i].points( rect_points );
////            for ( int j = 0; j < 4; j++ )
////            {
////                line( img, rect_points[j], rect_points[(j+1)%4], color );
////            }
////        }
////    Scalar color(rand()%255,rand()%256,rand()%256);
////    Rect rect(x,y,w,h);
////    rectangle(img, rect, color,LINE_8);
////    thet=(thet/CV_PI)*180;
//    
//    waitKey(0);
//    
//    //contours of a particular region
//    return 0;
//}
