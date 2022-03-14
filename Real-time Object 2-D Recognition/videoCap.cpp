/*
//  videoCap.cpp
 Corresponds to object recognition on live video.
 This file has 5 keypress actions avaialable for the user with the following functions:
 1. t: simple classifier training mode. Captures a frame on keypress, computes its features and appends to csv file
 2. o: simple object recognition mode. Computes label for object by assigning the label with the least distance
 3. k: knn classifier object recognition on live video. Assumes database contains multiple entries for each object
 4. q: exit process"
*/

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Tasks.h"
#include "csv_util.h"
#include "vid_tasks.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;
    vector<float> variance;
    vector<vector<float>> training_model_data;
    vector<char*> img_names;
    int count = 1;
    int k = 0;
    map<char*,int> mp;
    
    
    cout << "Keypress options: " << endl;
    cout << "t: simple classifier training mode" << endl;
    cout << "o: simple object recognition mode" << endl;
    cout << "k: knn classifier object recognition" << endl;
    cout << "q: exit process" << endl;
    
    // open the video device
    capdev = new cv::VideoCapture(1);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    cv::namedWindow("Video", 1); // identifies a window
    cv::Mat frame,filter;
    int flag = 0;
    
    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
            
        frame.copyTo(filter);
        //for user initiated training
        //capture one frame and send to training_vid function
        
        char key = cv::waitKey(10);
        
        //apply correspinding keypress actions on live video
        if(flag == 1){
            show_features(filter);
        }
        if(flag == 2){
            char* assigned_label;
            single_object_classifier(filter, training_model_data, variance, img_names, assigned_label);
        }
        if(flag == 3){
           knn_class_vid(filter, k, training_model_data, img_names, variance,mp);
        }
        
        cv::imshow("Video", filter);
        
        //keypreses perform the following functions
/*------------------------------------------------------------------------------------------------------------------------------------------
                                                      USER INITIATED TRAINING MODE
------------------------------------------------------------------------------------------------------------------------------------------*/
            
        //capture single frame
        if (key == 't'){
            string str = "/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/Resources/vid_training/frame_cap/vid_image"+to_string(count)+".jpg";
            imwrite(str, frame);
            count++;
            Mat img = imread(str);
            string label;
            cout << "Enter label" << endl;
            cin >> label;
            char* char_arr;
            string str_obj(label);
            char_arr = &str_obj[0];
            training_vid(img,char_arr);
            flag = 1;
        }
            
/*------------------------------------------------------------------------------------------------------------------------------------------
                                                      SIMPLE OBJECT RECOGNITION
------------------------------------------------------------------------------------------------------------------------------------------*/
        if(key == 'o'){
            //normalize existing feature vectors in the file
            read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/vid_model.csv", img_names, training_model_data, 0);
            cal_variance(variance, training_model_data);
            flag = 2;
        }
            
        
/*------------------------------------------------------------------------------------------------------------------------------------------
                                                        KNN CLASSSIFIER DETECTION MODE
------------------------------------------------------------------------------------------------------------------------------------------*/
        if(key == 'k'){
            cout << "Enter k value for knn classifier: " << endl;
            cin >> k;
            flag = 3;
        }
        
/*------------------------------------------------------------------------------------------------------------------------------------------
                                                            EXIT
------------------------------------------------------------------------------------------------------------------------------------------*/
        if( key == 'q') {
            break;
        }
        
    }
    
    delete capdev;
    return(0);
}

