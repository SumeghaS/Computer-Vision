//
//  main.cpp
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
#include "Tasks.h"
#include <iterator>
#include <map>
#include "csv_util.h"

using namespace cv;
using namespace std;

int main(int, char**) {
    
/*-----------------------------------------------------------------------------------------------------
                                        BASIC TRAINING MODEL
------------------------------------------------------------------------------------------------------*/
    
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
//    map<string,feature_data> model; //for just 1 feature vector/object
    
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
            Mat img,tmp,binary;
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
//            imshow("original image",img);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            
            //threshholding
            thresholding(img, tmp);
//            imshow("binary image",tmp);
            
            //clean up binary image
            grassfire_shrinking(tmp, binary, 2);
            grassfire_growing(tmp, binary,30);
            grassfire_shrinking(tmp, binary, 20);
//            imshow("cleanup",tmp);

//            segmentation using connected components
            Mat labels;
            Mat stats;
            Mat centroids;
            region_mapping(img, tmp, labels, stats, centroids);

            //segmentation using contours
            vector<vector<Point>> contours;
            oriented_bb(img, tmp, contours);
            vector<float> img_data;
            
            //seperates major regions and calculates features
            major_regions(tmp,contours,img_data);
            
            string img_name = dp->d_name;
            size_t lastindex = img_name.find_last_of(".");
            string rawname = img_name.substr(0, lastindex);
            
            char* char_arr;
            string str_obj(rawname);
            char_arr = &str_obj[0];
            cout << char_arr;
            
            //adding features to CSV file
            append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/training_model.csv", char_arr, img_data, 0);
     
//            waitKey(0);
        }
        
    }
    
    vector<vector<float>> training_model_data;
    vector<char*> img_names;
    
    //scaled euclidian normalization
    read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/training_model.csv", img_names, training_model_data, 0);
    
    vector<float> variance;
    for(int i=0;i<training_model_data[0].size();i++){//col
        double mean=0;
        for(int j=0;j<training_model_data.size();j++){//row
            mean+=training_model_data[j][i];
            mean/=training_model_data.size();
        }
        double var=0;
        for(int j=0;j<training_model_data.size();j++){//row
            var+=(training_model_data[j][i]-mean)*(training_model_data[j][i]-mean);
        }
        var/=training_model_data.size()-1;
        variance.push_back(var);
    }
    
    for(int i=0;i<variance.size();i++){
        cout << variance[i] << endl;
    }
    /*---------------------------------------------------------------------------------------------------
                                        USER INITIATED TRAINING/OBJECT RECOGNITION
     ---------------------------------------------------------------------------------------------------*/
    
    char mode;
    cout << "Press t for training mode" << endl;
    cout << "Press o for object recognition" << endl;
    cin >> mode;
    
    
    //training mode
    if(mode == 't' || mode == 'T'){
        //training mode
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
        
        //thresholding
        thresholding(img, tmp);
        imshow("binary image",tmp);
        
        //clean up binary image
        grassfire_shrinking(tmp, binary, 2);
        grassfire_growing(tmp, binary,30);
        grassfire_shrinking(tmp, binary, 20);
        imshow("cleanup",tmp);
        
        //segmentation using connected components
        Mat labels;
        Mat stats;
        Mat centroids;
        region_mapping(img, tmp, labels, stats, centroids);
        
        //segmentation using contours
        vector<vector<Point>> contours;
        oriented_bb(img, tmp, contours);
        
        vector<float> img_data;
        
        //seperates major regions and calculates features
        major_regions(tmp,contours,img_data);
        
        char* char_arr;
        string str_obj(label);
        char_arr = &str_obj[0];
//        cout << char_arr;
        
        //adds features to csv file
        append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/training_model.csv", char_arr, img_data, 0);
//        waitKey(0);
    }
    else{
        //object recognition mode
        
        //single object recognition
        
        string img_name;
        Mat img,tmp,binary;
        cout << "Enter image name" << endl;
        cin >> img_name;
        
        img = imread(img_name,IMREAD_COLOR);
        if(img.empty())
        {
            cout << "Could not access the image" << endl;
            return 1;
        }
        
        //thresholding
        thresholding(img, tmp);
        imshow("binary image",tmp);
        
        //clean up binary image
        grassfire_shrinking(tmp, binary, 2);
        grassfire_growing(tmp, binary,30);
        grassfire_shrinking(tmp, binary, 20);
        imshow("cleanup",tmp);
        
        //segmentation using connected components
        Mat labels;
        Mat stats;
        Mat centroids;
        region_mapping(img, tmp, labels, stats, centroids);
        
        //segmentation using contours
        vector<vector<Point>> contours;
        oriented_bb(img, tmp, contours);
        
        vector<float> img_data;
        
        //seperates major regions and calculates features
        major_regions(tmp,contours,img_data);
        
        //compare features to existing values in csv file and assign label
            //scaled euclidian distance metric
        //vector<float> img_data;vector<float> variance; vector<vector<float>> training_model_data;
        double dist = 0; //final dist vector weighted over all features
        double d = 0;
        for(int i=0;i<training_model_data.size();i++){ //rows(each image)
            for(int j=0;j<training_model_data[0].size();j++){//cols, each feature vector
               double d_temp= (training_model_data[i][j]-img_data[j])*(training_model_data[i][j]-img_data[j]);
                d_temp/=variance[j];
                d+=d_temp;
            }
        }
        d/=img_data.size();
        
        
        //multiple object recognition
        
    }
    return 0;
}

