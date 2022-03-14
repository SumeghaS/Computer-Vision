/*
Tasks.h
 Contains tasks corresponding to object recognition in images. Most of them are used for object recognition on live video too.
 Further details of functions are given above each of them.
*/
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

//datatype to store final distances and the corresponding object, used to assign labels
struct dist_metr_data{
    double dist;
    char* label;
};

//global vector to store the centroid of all major regions to put label text
vector<Point2f> label_coordinates;

//for quicksort
int Partition(vector<dist_metr_data> &v, int start, int end){
    
    int pivot = end;
    int j = start;
    for(int i=start;i<end;++i){
        if(v[i].dist<v[pivot].dist){
            swap(v[i],v[j]);
            ++j;
        }
    }
    swap(v[j],v[pivot]);
    return j;
    
}

//quicksort algorithm, sorts according to dist in the dist_metr_data data type
void Quicksort(vector<dist_metr_data> &v, int start, int end ){
    
    if(start<end){
        int p = Partition(v,start,end);
        Quicksort(v,start,p-1);
        Quicksort(v,p+1,end);
    }
    
};

//Thresholding function to obtain binary image with foreground set to white and background to black
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

//Growing function implemented using grassfire transform
//img is the binary image obtained after thresholding, binary is the grassfire transform matrix, final changes are made to img
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

//Shrinking function implemented using grassfire transform
//img is the binary image obtained after thresholding(and growing if done), binary is the grassfire transform matrix, final changes are made to img
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

//calculates regions in binary image, using connectedComponents with stats and makes bounding boxes
//img is the input image from imread, th is the binary and cleared image
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
//        rectangle(connected_regions, rect, color,LINE_8);
        rectangle(img, rect, color,LINE_8);
    }
    //    imshow("regions", connected_regions);
    return 0;
}

//calculates all regions in binary image using contours and makes oriented bounded boxes
//img is the input image from imread, binary is the binary and cleared image
int oriented_bb(Mat &img, Mat &binary, vector<vector<Point>> &contours){
    
    //contours of image
    findContours( binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    //rotated rectangle coordinates for each contour
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
//            line( img, rect_points[j], rect_points[(j+1)%4], color,LINE_8 );
        }
    }
    //    imshow( "All contours", drawing );
    return 0;
}

//function to find only the major/required regions from all the regions obtained using contours and further find their feature vectors
//conditions: contour size > n & ignore those with box along edges/or starting point
//features: percentage of oriented bounding filled, huMoments 0-5
//img is the input image from imread, binary is the binary and cleared image
//takes ocuntours vector generated from the oriented _bb function
int major_regions(Mat &img, Mat &binary, vector<vector<Point>> &contours, vector<float> &img_data){
    vector<RotatedRect> minRect( contours.size() );
    Mat drawing = Mat::zeros( binary.size(), CV_8UC3 );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
    }
    for( size_t i = 0; i< contours.size(); i++ ) //traverse over all contours/regions
    {   int flag = 1; //assume major region
        double area = minRect[i].size.height*minRect[i].size.width;
        if(area > 360000){
            Point2f rect_points[4];
            minRect[i].points( rect_points );
            for ( int j = 0; j < 4; j++ )
            {   //if any coordinates lie close to the edges
                if(rect_points[j].x<10 || rect_points[j].y<10 || rect_points[j].x>binary.cols-10 || rect_points[j].y>binary.rows-10){
                    flag = 0; //not a major region
                    break;
                }
            }
            
            if(flag == 1){
                //calculate features
                
                //oriented bb & ratio
                double bb_ratio=0;
                Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
                Scalar color2 = Scalar( rand()%255,rand()%255,rand()%255 );
//                drawContours( img, contours, (int)i, color,LINE_8 );
                Point2f rect_points[4];
                minRect[i].points( rect_points );
                
                for ( int j = 0; j < 4; j++ )
                {
                    line( img, rect_points[j], rect_points[(j+1)%4], color,LINE_8 );
                }
                
                
                bb_ratio = minRect[i].size.width/minRect[i].size.height;
                
                
                // central axis and centroid
                Moments m = moments(contours[i]);
                double cx = m.m10/m.m00;
                double cy = m.m01/m.m00;
                Point2f p;
                p.x = cx;
                p.y = cy;
                label_coordinates.push_back(p); //send coordinates of major region to global vector
                
                //calculating central axis angle
                double internal = (2*m.mu11)/(m.mu20-m.mu02);
                double thet = 0.5*atan(internal);
               
                //draw central axis on image
                double cx2 = cx+500*cos(thet);
                double cy2 = cy+500*sin(thet);
                double cx3 = cx+500*cos(thet+CV_PI);
                double cy3 = cy+500*sin(thet+CV_PI);
                
                //uncomment to draw line
//                line(img, Point(cx3,cy3), Point(cx2,cy2), color2,2,LINE_8);
                
                
                //1. Hu moment invariants
                double huMoments[7];
                HuMoments(m, huMoments);
                //transform to comparable scale
                for(int i = 0; i < 7; i++) {
                    huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                }
                
                
                //2. percentage filled
                double perc = contourArea(contours[i])/area;
                
                //all features of the region
                img_data.push_back(perc);
                img_data.push_back(huMoments[0]);
                img_data.push_back(huMoments[1]);
                img_data.push_back(huMoments[2]);
                img_data.push_back(huMoments[3]);
                img_data.push_back(huMoments[4]);
                img_data.push_back(huMoments[5]);
                
            }
        }
    }
    
//    imshow( "Major regions", img );
//    waitKey(0);
    
    return 0;
}

//function that computes the final feature vectors of all regions in an image using above functions
//Takes an image, processes it to a binary image, and stores its features in vector img_data
int cal_feature_vector(Mat &img, vector<float> &img_data){
    Mat tmp,binary;
    //thresholding
    thresholding(img, tmp);
    //    imshow("binary image",tmp);
    
    //    clean up binary image
    grassfire_shrinking(tmp, binary, 2);
    grassfire_growing(tmp, binary,30);
    grassfire_shrinking(tmp, binary, 20);
    //imshow("cleanup",tmp);
    
    //    uncomment to use connectedComponenets instead of contours
    //    segmentation using connected components
    //    Mat labels;
    //    Mat stats;
    //    Mat centroids;
    //    region_mapping(img, tmp, labels, stats, centroids);
    
    //segmentation using contours
    vector<vector<Point>> contours;
    oriented_bb(img, tmp, contours);
    
    //seperates major regions and calculates features
    major_regions(img,tmp,contours,img_data);
    
    return 0;
}

//function that generates a basic training model based on a directory containing a single image of each object
//calculates feature vector for each image in the directory and stores them in a csv file
//assumes that each image only has a single major region and extracts object name from image name
int model(){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    
    //traverse through all files
    cout << "Enter directory name";
    cin >> dirname;
    printf("Processing directory %s\n", dirname );
    
    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {
        
        if( strstr(dp->d_name, ".jpg") ||
           strstr(dp->d_name, ".png") ||
           strstr(dp->d_name, ".ppm") ||
           strstr(dp->d_name, ".tif") ) {
            Mat img;
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            vector<float> img_data;
            
            //calculate feature vectors of major regions in image
            cal_feature_vector(img, img_data);
            
            string img_name = dp->d_name;
            size_t lastindex = img_name.find_last_of(".");
            string rawname = img_name.substr(0, lastindex);
            
            char* char_arr;
            string str_obj(rawname);
            char_arr = &str_obj[0];
            
            //writing feature vectors to CSV file
            append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", char_arr, img_data, 0);
        }
    }
    return 0;
}

//function that generates a training model for a knn classifier with multiplle entries for each object
//assumes that each image only has a single major region and takes the object name as an input
//called to add multiple entries in training data each object
int model_knn(char* img_name){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    
    //traverse through all files
    cout << "Enter directory name";
    cin >> dirname;
    printf("Processing directory %s\n", dirname );
    
    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {
        
        if( strstr(dp->d_name, ".jpg") ||
           strstr(dp->d_name, ".png") ||
           strstr(dp->d_name, ".ppm") ||
           strstr(dp->d_name, ".tif") ) {
            Mat img;
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            vector<float> img_data;
            cal_feature_vector(img, img_data);
            
            //adding feature vector to CSV file
            append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_name, img_data, 0);
        }
        
    }
    return 0;
}

//user initiated training model
//takes the number of images/objects user wants to train as an input and adds their feature vectors to training data model
int user_training(int num_t){
 
    //training mode
    for(int num = 0;num<num_t;num++){
        string img_name;
        string label;
        Mat img,tmp,binary;
        cout << "Enter image name" << endl;
        cin >> img_name;
        cout << "Enter label" << endl;
        cin >> label;
        img = imread(img_name,IMREAD_COLOR);
        if(img.empty())
        {
            cout << "Could not access the image" << endl;
            return 1;
        }
        
        vector<float> img_data;
        cal_feature_vector(img, img_data);
        char* char_arr;
        string str_obj(label);
        char_arr = &str_obj[0];

        
        //adding features to CSV file
        append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", char_arr, img_data, 0);
        
        
    }
    return 0;
}

//function to calculate variance of each feature in the training data to make the data homogeneous
//training_model_data contains the feature vectors from the csv fille
int cal_variance(vector<float> &variance, vector<vector<float>> &training_model_data){
    
    //if variance is non empty, clear and calculate again (in case new data was added)
    if(!variance.empty()){
        variance.clear();
    }
    
    
    for(int j=0;j<training_model_data[0].size();j++){//traversing through cols ie features
        //calculate mean of each feature
        double mean=0;
        for(int i=0;i<training_model_data.size();i++){//traversing through rows ie images
            mean+=training_model_data[i][j];
        }
        mean/=training_model_data.size(); //mean of 1 feature
        
        //calculate variance of each feature
        double var=0;
        for(int i=0;i<training_model_data.size();i++){//row
            var+=(training_model_data[i][j]-mean)*(training_model_data[i][j]-mean);
        }
        var/=training_model_data.size();
        variance.push_back(var);
    }

    return 0;
}

//function calculate scaled euclidian distance wrt to an object requiring labeling
int dist_metric(vector<float> &img_data,vector<dist_metr_data> &obj_recognition, vector<vector<float>> &training_model_data,vector<float> &variance, vector<char*> &img_names){
    
    //scaled euclidian distance metric
    double d = 0;
    for(int i=0;i<training_model_data.size();i++){ //rows(each image)
        dist_metr_data obj;
        obj.label = img_names[i];
        for(int j=0;j<training_model_data[0].size();j++){//cols, each feature vector
            double d_temp= (training_model_data[i][j]-img_data[j])*(training_model_data[i][j]-img_data[j]);
            d_temp/=variance[j];
            d+=d_temp;
        }
        //add distance metric and label in a vector
        d/=img_data.size();
        obj.dist = d;
        obj_recognition.push_back(obj);
    }
    
    return 0;
}

//simple classifier that computes distances from each object and gives the object the label with least distance
int single_object_classifier(Mat &img,vector<vector<float>> &training_model_data,vector<float> &variance, vector<char*> &img_names,char* assigned_label){
    label_coordinates.clear();
    vector<dist_metr_data> obj_recognition;
    vector<float> img_data;
    
    //calculate feature vector of unknown image
    cal_feature_vector(img, img_data);
    
    //compare features to existing values in csv file and assign label
    //scaled euclidian distance metric
    if(!img_data.empty()){
        dist_metric(img_data, obj_recognition, training_model_data, variance,img_names);
        //sort vector
        Quicksort(obj_recognition, 0, obj_recognition.size()-1);
        //output closest match
//        for(int i=0;i<obj_recognition.size();i++){
//            cout << obj_recognition[i].label << ": " << obj_recognition[i].dist << endl;
//        }
        strcpy(assigned_label,obj_recognition[0].label);
//        cout << assigned_label << endl;
        putText(img,obj_recognition[0].label,/*Point(500,500)*/label_coordinates.at(0),FONT_HERSHEY_COMPLEX,7,cv::Scalar(255,0,0),2,false);
//        imshow("labeled object", img);
//        waitKey(0);
    }
    else{
        cout << "Couldn't detect major region" << endl;
    }
    
    return 0;
}

//knn classifier
//asks the user to add multiple images for each object already present in single entry model
//computes distances from each entry and takes sum of top k as the final distance from object
//assigns label with least distance
//can detect multiple objects simultaneously in the same image
int knn_classifier(Mat &img, int k){
    label_coordinates.clear();
    vector<vector<float>> training_model_data;
    vector<char*> img_names;
    vector<dist_metr_data> obj_recognition;  //final distance metric vector
    read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_names, training_model_data, 0);
    
    //map keeps track of
    // 1. if we have multiple entries of the object or not
    // 2. if distance has been computed for all entries of one object
    map<char* , int> mp;
    
//------------------------------------------------------------------------------------------------------
    //1. Calculate features of unkown image
    vector<float> img_data;
    cal_feature_vector(img, img_data);
//------------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------------
    //2. ask for multiple entries for each existing label, compute new feature vectors, add them to csv file
    if(!img_data.empty()){
        int reg=0;           //for if theere are multiple major regions/objects in an image
        while(reg<img_data.size()){
            int temp_reg = reg;
            
            for(int i=0;i<img_names.size();i++){
                auto itr = mp.find(img_names[i]);
                if(itr == mp.end()){//object doesn't have multiple entries yet
                    
//                    cout << "Enter images for object: " << img_names[i] << endl;
//                    model_knn(img_names[i]);
                    
                    //add object to map
                    mp.insert({img_names[i],0});
                }
                //The map now contains all the objects with a 0 value
            }
//------------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------------
            //3. Calculate variance after adding all new values
            vector<float> variance;
            training_model_data.clear();
            img_names.clear();
            read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_names, training_model_data, 0);
            cal_variance(variance, training_model_data);
//------------------------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------------------
            //4. Calculate distance from each entry of 1 object and pick top k, add them and add the final d value to final obj recognition vector
            for(int i=0;i<img_names.size();i++){
                int j=0;
                int flag=0;
                vector<dist_metr_data> label_dist;  //vector for all distances of each object
                auto itr = mp.begin();
                while(itr!=mp.end()){
                    if(strcmp(itr->first,img_names[i])==0 && itr->second == 0){
                        flag = 1;
                        break;
                    }
                    itr++;
                }
                
                //if value in map is false i.e distance has not been computed for that object yet
                if(flag==1){
                    while(j<img_names.size()){
                        if(strcmp(img_names[j],img_names[i])==0){
                            //going over all entries with same labels
                            double d = 0;
                            dist_metr_data obj;
                            obj.label = img_names[i];
                            for(int col=0;col<training_model_data[0].size();col++){//cols, each feature vector
                                double d_temp= (training_model_data[j][col]-img_data[temp_reg])*(training_model_data[j][col]-img_data[temp_reg]);
                                d_temp/=variance[col];
                                d+=d_temp;
                                temp_reg++;
                            }
                            d/=img_data.size(); //equal weight for all features
                            obj.dist = d;
                            label_dist.push_back(obj); //contains dist for all entries of object
                            temp_reg=reg;
                        }
                        j++;
                    }
                    
                    //sort all entries for one object
                    Quicksort(label_dist, 0, label_dist.size()-1);
                    
                    //            cout << "Distances for label: " << img_names[i] << endl;
                    //            for(int iter = 0; iter<label_dist.size();iter++){
                    //                cout << label_dist[iter].dist << ", ";
                    //            }
                    
                    double d_final = 0;
                    if(label_dist.size()>=k){
                        //pick top k
                        for(int t=0;t<k;t++){
                            d_final+=label_dist[t].dist;
                        }
                    }
                    else{ //incase no major regions could be detected in the input images
                        d_final = k*label_dist[0].dist;
                    }
                    
                    //add to final distance metric vector
                    dist_metr_data obj_f;
                    obj_f.label = img_names[i];
                    obj_f.dist = d_final;
                    obj_recognition.push_back(obj_f);
                    
                    //update value to true in map
                    itr->second = 1;
                }
                
            }
            //iteration over one major region is over, move to next
            reg = reg+training_model_data[0].size();
            
            //set map values to 0 again for the next major region
            for(auto itr = mp.begin();itr!=mp.end();itr++){
                itr->second = 0;
            }
        }
//------------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------------
        //5. sort final distance vector and label image
        //if there are multiple major regions, sort object distances for each region
        int begin_sort = 0;
        while(begin_sort+mp.size()<=obj_recognition.size()){
            Quicksort(obj_recognition, begin_sort, begin_sort+mp.size()-1);
            begin_sort +=mp.size();
        }
        
        //if unknown object
        int label_index = 0;
        int unk_index = 0;
        string unk_label;
        while(label_index<obj_recognition.size()){
            char train_flag_unk;
            if(obj_recognition[label_index].dist>0.5){
                //train here only, we already have the feature vectors.
                cout << "Unkown object, train? y/n" << endl;
                cin >> train_flag_unk;
                vector<float> unk_feat;
                if(train_flag_unk == 'y'){
                    //ask for label
                    cout << "Label for unkown object?" << endl;
                    cin >> unk_label;
                    for(int un_i=unk_index;un_i<unk_index+training_model_data[0].size();un_i++){
                        unk_feat.push_back(img_data[un_i]);
                    }
                    
                    //append to csv
                    char* char_arr;
                    string str_obj(unk_label);
                    char_arr = &str_obj[0];
                    append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", char_arr, unk_feat, 0);
                }
                
            }
            unk_index+=training_model_data[0].size();
            label_index+=mp.size();
        }

        
        //label each major region
        label_index=0;
        int label_c=0;
        while(label_index<obj_recognition.size()){

            putText(img,obj_recognition[label_index].label,/*Point(x,y)*/label_coordinates[label_c],FONT_HERSHEY_COMPLEX,5,cv::Scalar(255,0,0),2,false);
//            strcpy(assigned_label, obj_recognition[0].label);
            label_index+=mp.size();
            label_c++;
        }
        imshow("labeled object", img);
        waitKey(0);
    }
    else{
        cout << "Couldn't detect major region" << endl;
    }
    
    return 0;
}

//knn classifier assuming training data already contains multiple entries of each object
int knn_without_training(Mat &img, int k, char* assigned_label,vector<std::vector<float>> &training_model_data,vector<char*> &img_names, vector<float> &variance){
    label_coordinates.clear();
    vector<dist_metr_data> obj_recognition;  //final distance metric vector
    map<char* , int> mp;
    
//------------------------------------------------------------------------------------------------------
    //1. Calculate features of unkown image
    vector<float> img_data;
    cal_feature_vector(img, img_data);
//------------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------------
    //2. initialize map
    
    if(!img_data.empty()){
        int reg=0;           //for if theere are multiple major regions/objects in an image
        while(reg<img_data.size()){
            int temp_reg = reg;
            for(int i=0;i<img_names.size();i++){
                auto itr = mp.find(img_names[i]);
                if(itr == mp.end()){//object doesn't have multiple entries yet
                    //add object to map
                    mp.insert({img_names[i],0});
                }
                //The map now contains all the objects with a 0 value
            }
            //------------------------------------------------------------------------------------------------------
            
            //-----------------------------------------------------------------------------------------------------
            //3. Calculate distance from each entry of 1 object and pick top k, add them and add the final d value to final obj recognition vector
            for(int i=0;i<img_names.size();i++){
                int j=0;
                int flag=0;
                vector<dist_metr_data> label_dist;  //vector for all distances of each object
                auto itr = mp.begin();
                while(itr!=mp.end()){
                    if(strcmp(itr->first,img_names[i])==0 && itr->second == 0){
                        flag = 1;
                        break;
                    }
                    itr++;
                }
                
                //if value in map is false i.e distance has not been computed for that object yet
                if(flag==1){
                    while(j<img_names.size()){
                        if(strcmp(img_names[j],img_names[i])==0){
                            //going over all entries with same labels
                            double d = 0;
                            dist_metr_data obj;
                            obj.label = img_names[i];
                            for(int col=0;col<training_model_data[0].size();col++){//cols, each feature vector
                                double d_temp= (training_model_data[j][col]-img_data[temp_reg])*(training_model_data[j][col]-img_data[temp_reg]);
                                d_temp/=variance[col];
                                d+=d_temp;
                                temp_reg++;
                            }
                            d/=img_data.size(); //equal weight for all features
                            obj.dist = d;
                            label_dist.push_back(obj); //contains dist for all entries of object
                            temp_reg=reg;
                        }
                        j++;
                    }
                    
                    //sort all entries for one object
                    Quicksort(label_dist, 0, label_dist.size()-1);
                    
                    //            cout << "Distances for label: " << img_names[i] << endl;
                    //            for(int iter = 0; iter<label_dist.size();iter++){
                    //                cout << label_dist[iter].dist << ", ";
                    //            }
                    
                    double d_final = 0;
                    if(label_dist.size()>=k){
                        //pick top k
                        for(int t=0;t<k;t++){
                            d_final+=label_dist[t].dist;
                        }
                    }
                    else{ //incase no major regions could be detected in the input images
                        d_final = k*label_dist[0].dist;
                    }
                    
                    //add to final distance metric vector
                    dist_metr_data obj_f;
                    obj_f.label = img_names[i];
                    obj_f.dist = d_final;
                    obj_recognition.push_back(obj_f);
                    
                    //update value to true in map
                    itr->second = 1;
                }
                
            }
            //iteration over one major region is over, move to next
            reg = reg+training_model_data[0].size();
            
            //set map values to 0 again for the next major region
            for(auto itr = mp.begin();itr!=mp.end();itr++){
                itr->second = 0;
            }
        }
     
        
//------------------------------------------------------------------------------------------------------
    
//------------------------------------------------------------------------------------------------------
        //4. sort and label
        //if there are multiple major regions, sort object distances for each region
        int begin_sort = 0;
        while(begin_sort+mp.size()<=obj_recognition.size()){
            Quicksort(obj_recognition, begin_sort, begin_sort+mp.size()-1);
            begin_sort +=mp.size();
        }
        
//        //if unknown object
        int label_index = 0;
        int unk_index = 0;
        string unk_label;
        while(label_index<obj_recognition.size()){
            char train_flag_unk;
            if(obj_recognition[label_index].dist>0.5){
                //train here only, we already have the feature vectors.
                cout << "Unkown object, train? y/n" << endl;
                cin >> train_flag_unk;
                vector<float> unk_feat;
                if(train_flag_unk == 'y'){
                    //ask for label
                    cout << "Label for unkown object?" << endl;
                    cin >> unk_label;
                    for(int un_i=unk_index;un_i<unk_index+training_model_data[0].size();un_i++){
                        unk_feat.push_back(img_data[un_i]);
                    }

                    //append to csv
                    char* char_arr;
                    string str_obj(unk_label);
                    char_arr = &str_obj[0];
                    append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", char_arr, unk_feat, 0);
                }

            }
            unk_index+=training_model_data[0].size();
            label_index+=mp.size();
        }
        
//        label each major region
        label_index=0;
        int label_c=0;
        
        while(label_index<obj_recognition.size()){
            strcpy(assigned_label,obj_recognition[0].label);
            putText(img,obj_recognition[label_index].label,/*Point(x,y)*/label_coordinates[label_c],FONT_HERSHEY_COMPLEX,5,cv::Scalar(255,0,0),2,false);
            label_index+=mp.size();
            label_c++;
        }
        imshow("labeled object", img);
        waitKey(0);
    }
    else{
        
        cout << "Couldn't detect major region" << endl;
    }
    
    return 0;
}

//function to calculate confusion matrix according to both simple classifier and knn classifier
int confusion_matrix(vector<char*> &filenames, vector<std::vector<float>> &data, vector<float> &variance){
    vector<char*> img_names;
    char* assigned_label;
    int k;
    int class_flag = 0;
    
    cout << "single classifier or knn? 0/1" << endl;
    cin >> class_flag;
    if(class_flag == 1){
        cout << "K value for knn? " << endl;
        cin >> k;}
    
    for(int i=0;i<filenames.size();i++){
        int flag = 0;
        //traverse through vector img_names and see if already exists
        for(int j=0;j<img_names.size();j++){
            if(strcmp(img_names[j],filenames[i])==0){
                flag = 1;
                break;
            }
        }
        if(flag == 0){
            //does not exist in vector
            img_names.push_back(filenames[i]);
        }
    }
    
    
    for(int i=0;i<img_names.size();i++){
        vector<float> truth_vector;
        
        //initialise truth vector to 0
        for(int j=0;j<img_names.size();j++){
            truth_vector.push_back(0);
        }
        
        char dirname[256];
        char buffer[256];
        DIR *dirp;
        struct dirent *dp;
        
        //test images for each object
        cout << "Enter directory name for object " << img_names[i];
        cin >> dirname;
        printf("Processing directory %s\n", dirname );
        
        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
        while( (dp = readdir(dirp)) != NULL ) {
            
            if( strstr(dp->d_name, ".jpg") ||
               strstr(dp->d_name, ".png") ||
               strstr(dp->d_name, ".ppm") ||
               strstr(dp->d_name, ".tif") ) {
                Mat img;
                char* assigned_label;
                strcpy(buffer, dirname);
                strcat(buffer, "/");
                strcat(buffer, dp->d_name);
                
                img = imread(buffer, IMREAD_COLOR);
                
                if(img.empty())
                {
                    cout << "Could not access the image" << endl;
                    return 1;
                }
                
                if(class_flag == 0){
                    single_object_classifier(img, data, variance, filenames, assigned_label);
                    cout << assigned_label << endl;
                }
                //run knn and find label
                else{
                    string temp = "";
                    char* char_arr;
                    string str_obj(temp);
                    char_arr = &str_obj[0];
                    knn_classifier(img, k);
//                    knn_without_training(img, k, assigned_label, data, filenames, variance);
//                    cout << assigned_label << endl;
                }
                
                if(strcmp(assigned_label,"")!=0){
                    //go through img_names vector and ++ in truth vector at that index
                    for(int j=0;j<img_names.size();j++){
                        if(strcmp(assigned_label,img_names[j])==0){
                            truth_vector.at(j) = truth_vector.at(j)+1;
                        }
                    }
                }
            }
            
        }
        append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/confusion_matrix.csv", img_names[i], truth_vector, 0);
    }
    
    
    return 0;
}

#endif /* Tasks_h */
