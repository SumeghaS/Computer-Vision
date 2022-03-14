/*
  main.cpp
 Corresponds to object recognition on images.
 This file has 5 actions avaialable for the user with the following functions:
 1. 'b': Basic training model. Asks for a directory with a single image corresponding to an object and stores the feature vectors for each
 2. 't': User initiated training mode. Asks the user to enter an image path and label and stores the feature vectors
 3. 'o': object recognition against the training model data obtained above using a simple classifier
 4. 'k': object recognition using knn classifier. Asks user for a value of k.
 5. 'c': to build a confusion matrix using either simple classifier or knn classifier
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
#include "Tasks.h"
#include <iterator>
#include <map>
#include "csv_util.h"

using namespace cv;
using namespace std;

int main(int, char**) {
    
    cout << "Press b for basic training mode" << endl;
    cout << "Press t for training mode" << endl;
    cout << "Press o for single object recognition" << endl;
    cout << "Press k for knn object recognition" << endl;
    cout << "Press c for object recognition" << endl;
    
    char mode;
    cin >> mode;
    vector<float> variance;
    vector<vector<float>> training_model_data;
    vector<char*> img_names;
    
/*-------------------------------------------------------------------------------------------------------------------------------------------------
                                                            BASIC TRAINING MODEL
-------------------------------------------------------------------------------------------------------------------------------------------------*/
    if(mode == 'b' || mode == 'B'){
        model();
    }

/*------------------------------------------------------------------------------------------------------------------------------------------------
                                                            USER INITIATED TRAINING
------------------------------------------------------------------------------------------------------------------------------------------------*/
    else if(mode == 't' || mode == 'T'){
    int num_t = 0;
    cout << "Enter number of objects for training" << endl;
    cin >> num_t;
        user_training(num_t);
    }
    
/*-------------------------------------------------------------------------------------------------------------------------------------------------
                                                    OBJECT RECOGNITION USING SIMPLE CLASSIFIER
-------------------------------------------------------------------------------------------------------------------------------------------------*/
    else if(mode == 'o' || mode == 'O'){
        
        //finding variance for objects in the training model
        read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_names, training_model_data, 0);
        cal_variance(variance, training_model_data);
        
        //comparision image
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
        char* assigned_label;
        single_object_classifier(img, training_model_data, variance, img_names, assigned_label);
    }
        
/*-------------------------------------------------------------------------------------------------------------------------------------------------
                                                    OBJECT RECOGNITION USING KNN CLASSIFIER
-------------------------------------------------------------------------------------------------------------------------------------------------*/
    else if(mode == 'k' || mode == 'K'){
        
        //finding variance for objects in the training model
        read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_names, training_model_data, 0);
        cal_variance(variance, training_model_data);
        
        //comparision image
        string img_name;
        Mat img,tmp,binary;
        cout << "Enter image name" << endl;
        cin >> img_name;
        
        //        img = imread(img_name,IMREAD_COLOR);
        img = imread(img_name,IMREAD_COLOR);
        if(img.empty())
        {
            cout << "Could not access the image" << endl;
            return 1;
        }
        char* assigned_label;
        
        int k;
        cout << "K value for KNN classifier: " << endl;
        cin >> k;
//        knn_without_training(img, k, assigned_label, training_model_data, img_names, variance);
        knn_classifier(img, k);
    }

/*-----------------------------------------------------------------------------------------------------------------------------------------------
                                                            CONFUSION MATRIX
-----------------------------------------------------------------------------------------------------------------------------------------------*/
    else if(mode == 'c' || mode == 'C'){
        
        read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/training_model.csv", img_names, training_model_data, 0);
        cal_variance(variance, training_model_data);
        confusion_matrix(img_names, training_model_data, variance);
    }

    return 0;

}
