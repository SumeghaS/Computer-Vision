/*
 vid_tasks.h
 Contains additional functions specifically designed for use in object recognition on realtime video
 Further details of functions are given above each of them.
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <dirent.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iterator>
#include <map>

#ifndef vid_tasks_h
#define vid_tasks_h


//function to display major regions on live video
//img corresponds to each frame
int major_regions_show(Mat &img, Mat &binary){
    vector<vector<Point>> contours;
    
    findContours( binary, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<RotatedRect> minRect( contours.size() );
//    Mat drawing = Mat::zeros( binary.size(), CV_8UC3 );
    
    for( size_t i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( contours[i] );
    }
    for( size_t i = 0; i< contours.size(); i++ )
    {   int flag = 1;
        double area = minRect[i].size.height*minRect[i].size.width;
        if(area > 36000){
            Point2f rect_points[4];
            minRect[i].points( rect_points );
            for ( int j = 0; j < 4; j++ )
            {
                if(rect_points[j].x<10 || rect_points[j].y<10 || rect_points[j].x>binary.cols-10 || rect_points[j].y>binary.rows-10){
                    flag = 0;
                    break;
                }
            }
            if(flag == 1){
                //calculate features
                
                // oriented bb & ratio
                double bb_ratio=0;
                Scalar color = Scalar( rand()%255,rand()%255,rand()%255 );
                Scalar color2 = Scalar( rand()%255,rand()%255,rand()%255 );
                // drawContours( img, contours, (int)i, color,LINE_8 );
                // rotated rectangle
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
                double internal = (2*m.mu11)/(m.mu20-m.mu02);
                double thet = 0.5*atan(internal);
                
                //draw central axis
                double cx2 = cx+500*cos(thet);
                double cy2 = cy+500*sin(thet);
                double cx3 = cx+500*cos(thet+CV_PI);
                double cy3 = cy+500*sin(thet+CV_PI);
                line(img, Point(cx3,cy3), Point(cx2,cy2), color2,2,LINE_8);
                
            }
        }
    }
  
    return 0;
}

//function to display major regions and features computed above on live video
//takes frame as input
int show_features(Mat &img){
    Mat tmp,binary;
    //thresholding
    thresholding(img, tmp);
    //imshow("binary image",tmp);
        
    //clean up binary image
    grassfire_shrinking(tmp, binary, 2);
    grassfire_growing(tmp, binary,30);
    grassfire_shrinking(tmp, binary, 20);
    //imshow("cleanup",tmp);

    major_regions_show(img, tmp);
    
    return 0;
}

//user initiated training model on live video
//takes captureed video frame during keypress as input and label as input by the user
int training_vid(Mat &img, char* label){
    
    //training mode
    Mat tmp,binary;
    if(img.empty())
    {
        cout << "Could not access the image" << endl;
        return 1;
    }
    
    vector<float> img_data;
    cal_feature_vector(img, img_data);
    
    //adding features to CSV file
    append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/vid_model.csv", label, img_data, 0);
    
    return 0;
}

//function for object recognition using knn classifier on live vide
//assumes that the training data already contains multiple entries for each object
//also detects unkown objects and asks if the user wants to train them
//capable of detecting multiple objects simultaneously in realtime
int knn_class_vid(Mat &img, int k, vector<vector<float>> &training_model_data, vector<char*> &img_names, vector<float> &variance, map<char*,int> mp){
    label_coordinates.clear();
    training_model_data.clear();
    img_names.clear();
    variance.clear();
    read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/vid_model.csv", img_names, training_model_data, 0);
    cal_variance(variance, training_model_data);
    vector<dist_metr_data> obj_recognition;
    
    //1.fill map
    for(int i=0;i<img_names.size();i++){
        auto itr = mp.find(img_names[i]);
        if(itr == mp.end()){ //object not in map yet
            mp.insert({img_names[i],0});
        }
    }
    
    //2. calculate features of frame
    vector<float> img_data;
    cal_feature_vector(img, img_data);
    
    
    //3. Calculate distance from each entry of 1 object and pick top k, add them and add the final d value to final obj recognition vector
    if(!img_data.empty()){
        int reg=0;
        while(reg<img_data.size()){
            //after each region comparision, set all map values to 0;
            
            int temp_reg = reg;
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
                
                //if value in map is false
                if(flag==1){
                    while(j<img_names.size()){ //looping over all entries in db
                        if(strcmp(img_names[j],img_names[i])==0){
                            //going over all entries with same labels
                            double d = 0;
                            dist_metr_data obj;
                            obj.label = img_names[i];
                            for(int col=0;col<training_model_data[0].size();col++){//cols, each feature vector (7)
                                double d_temp= (training_model_data[j][col]-img_data[temp_reg])*(training_model_data[j][col]-img_data[temp_reg]);
                                d_temp/=variance[col];
                                d+=d_temp;
                                temp_reg++;
                            }
                            //temp_reg = 7;
                            d/=training_model_data[0].size(); //equal weight for all features
                            obj.dist = d;
                            label_dist.push_back(obj); //contains dist for all entries of object
                            temp_reg=reg;
                        }
                        j++;
                    }
                    //temp_reg = reg when it comes out of this loop
                    
                    //sort
                    Quicksort(label_dist, 0, label_dist.size()-1);
                    
                    //                            cout << "Distances for label: " << img_names[i] << endl;
                    //                            for(int iter = 0; iter<label_dist.size();iter++){
                    //                                cout << label_dist[iter].dist << ", ";
                    //                            }
                    
                    double d_final = 0;
                    if(label_dist.size()>=k){
                        //pick top k
                        
                        for(int t=0;t<k;t++){
                            d_final+=label_dist[t].dist;
                        }
                    }
                    else{
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
            
            //iteration over one major region is over
            reg = reg+training_model_data[0].size();
            
            //        set map values to 0 again
            for(auto itr = mp.begin();itr!=mp.end();itr++){
                itr->second = 0;
            }
            
        }
        
        //4. sort and label
        //sort over every map.size()
        int begin_sort = 0;
        while(begin_sort+mp.size()<=obj_recognition.size()){
            Quicksort(obj_recognition, begin_sort, begin_sort+mp.size()-1);
            begin_sort +=mp.size();
            
        }
        
        //if unknown object
        int label_index = 0;
//        int unk_index = 0;
//        string unk_label;
//        while(label_index<obj_recognition.size()){
//            char train_flag_unk;
//            if(obj_recognition[label_index].dist>0.5){
//                //train here only, we already have the feature vectors.
//                cout << "Unkown object, train? y/n" << endl;
//                cin >> train_flag_unk;
//                vector<float> unk_feat;
//                if(train_flag_unk == 'y'){
//                    //ask for label
//                    cout << "Label for unkown object?" << endl;
//                    cin >> unk_label;
//                    for(int un_i=unk_index;un_i<unk_index+training_model_data[0].size();un_i++){
//                        unk_feat.push_back(img_data[un_i]);
//                    }
//
//                    //append to csv
//                    char* char_arr;
//                    string str_obj(unk_label);
//                    char_arr = &str_obj[0];
//                    append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/CV3/CV3/models/vid_model.csv", char_arr, unk_feat, 0);
//                }
//
//            }
//            unk_index+=training_model_data[0].size();
//            label_index+=mp.size();
//        }
        
        
        label_index=0;
        int label_c=0;
        while(label_index<obj_recognition.size()){
            putText(img,obj_recognition[label_index].label,/*Point(x,y)*/label_coordinates[label_c],FONT_HERSHEY_COMPLEX,5,cv::Scalar(255,0,0),2,false);
            label_index+=mp.size();
            label_c++;
        }
        
        
    }
    else{
        cout << "Couldn't detect major region" << endl;
    }
    
    return 0;
}

#endif /* vid_tasks_h */
