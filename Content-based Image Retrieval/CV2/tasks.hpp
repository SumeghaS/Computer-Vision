/*
    tasks.hpp
    The first part of this file contains the core Tasks 1-5 and additional functions used for
    the same. [The laws filters have been added in the beginning as they are used in Task5]
    The second part contains the two extensions: Gabor and Laws filter
    Additional details have been given with the corresponding functions.
//  Created by Sumegha Singhania on 2/10/22.
*/


#ifndef tasks_hpp
#define tasks_hpp
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

//data type created to store distance metric and img_name in the same vector
struct sort_data_type{
    string img_name;
    float dist_metric;
};

int laws_f1(Mat &src, Mat &dst){
    Mat3s tmp,tmp1;
    src.copyTo(tmp);
    src.copyTo(tmp1);
//   L5*E5
    
//  L5: [1 4 6 4 1]
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c] + src.at<Vec3b>(i,j-1)[c]*4 + src.at<Vec3b>(i,j)[c]*6 + src.at<Vec3b>(i,j+1)[c]*4+src.at<Vec3b>(i,j+2)[c])/16;
            }
        }
    }
    
//    E5: [-1 -2 0 2 1]T
    for(int i=2;i<src.rows-2;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                tmp1.at<Vec3s>(i,j)[c] = (tmp.at<Vec3s>(i-2,j)[c]*-1 + tmp.at<Vec3s>(i-1,j)[c]*-2 + tmp.at<Vec3s>(i+1,j)[c]*2+tmp.at<Vec3s>(i+2,j)[c]*1)/3;
            }
        }
    }
    
    convertScaleAbs(tmp1,dst);
    return 0;
}

int laws_f2(Mat &src, Mat &dst){
    Mat3s tmp,tmp1;
    src.copyTo(tmp);
    src.copyTo(tmp1);
//   E5*S5
    
//  E5 = [-1 -2 0 2 1]
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c]*-1 + src.at<Vec3b>(i,j-1)[c]*-2 + src.at<Vec3b>(i,j+1)[c]*-2+src.at<Vec3b>(i,j+2)[c]*1)/3;
            }
        }
    }
    
//    S5 = [-1 0 2 0 -1]T
    for(int i=2;i<src.rows-2;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                tmp1.at<Vec3s>(i,j)[c] = (tmp.at<Vec3s>(i-2,j)[c]*-1 + tmp.at<Vec3s>(i,j)[c]*2 +tmp.at<Vec3s>(i+2,j)[c]*-1)/2;
            }
        }
    }
    convertScaleAbs(tmp1,dst);
    return 0;
}

int laws_f3(Mat &src, Mat &dst){
    Mat3s tmp,tmp1;
    src.copyTo(tmp);
    src.copyTo(tmp1);
//    R5*R5
    
    
//     R5 = [1 -4 6 -4 1]
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c] + src.at<Vec3b>(i,j-1)[c]*-4 + src.at<Vec3b>(i,j)[c]*6 + src.at<Vec3b>(i,j+1)[c]*-4+src.at<Vec3b>(i,j+2)[c])/8;
            }
        }
    }
    
//     R5 = [1 -4 6 -4 1]T
    for(int i=2;i<src.rows-2;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                tmp1.at<Vec3s>(i,j)[c] = (tmp.at<Vec3s>(i-2,j)[c] + tmp.at<Vec3s>(i-1,j)[c]*-4 + tmp.at<Vec3s>(i,j)[c]*6 +tmp.at<Vec3s>(i+1,j)[c]*-4+tmp.at<Vec3s>(i+2,j)[c])/8;
            }
        }
    }
    
    
    
    convertScaleAbs(tmp1,dst);
    return 0;
}

int laws_f4(Mat &src, Mat &dst){
    Mat3s tmp,tmp1;
    src.copyTo(tmp);
    src.copyTo(tmp1);
//    L5*S5
    
    //  L5: [1 4 6 4 1]
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c] + src.at<Vec3b>(i,j-1)[c]*4 + src.at<Vec3b>(i,j)[c]*6 + src.at<Vec3b>(i,j+1)[c]*4+src.at<Vec3b>(i,j+2)[c])/16;
            }
        }
    }
    
    //    S5 = [-1 0 2 0 -1]T
    for(int i=2;i<src.rows-2;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                tmp1.at<Vec3s>(i,j)[c] = (tmp.at<Vec3s>(i-2,j)[c]*-1 + tmp.at<Vec3s>(i,j)[c]*2 +tmp.at<Vec3s>(i+2,j)[c]*-1)/2;
            }
        }
    }
    convertScaleAbs(tmp1,dst);
    return 0;
    
    
}

//not used, only for experimentation
int laws_f5(Mat &src,Mat &dst){
    Mat3s tmp,tmp1;
    src.copyTo(tmp);
    src.copyTo(tmp1);
//    L5*R5
    
    //  L5: [1 4 6 4 1]
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c] + src.at<Vec3b>(i,j-1)[c]*4 + src.at<Vec3b>(i,j)[c]*6 + src.at<Vec3b>(i,j+1)[c]*4+src.at<Vec3b>(i,j+2)[c])/16;
            }
        }
    }
    
    //     R5 = [1 -4 6 -4 1]T
        for(int i=2;i<src.rows-2;i++){
            for(int j=0;j<src.cols;j++){
                for(int c=0;c<3;c++){
                    tmp1.at<Vec3s>(i,j)[c] = (tmp.at<Vec3s>(i-2,j)[c] + tmp.at<Vec3s>(i-1,j)[c]*-4 + tmp.at<Vec3s>(i,j)[c]*6 +tmp.at<Vec3s>(i+1,j)[c]*-4+tmp.at<Vec3s>(i+2,j)[c])/8;
                }
            }
        }    
        
        convertScaleAbs(tmp1,dst);
        return 0;
  
    
}

//finds 3d RGB histogram of image
int hist3d(Mat&img, float *hist3d, int Hsize){
    
    //initializee to zeros
    int divisor = 256/Hsize;
    for(int i=0;i<Hsize*Hsize*Hsize;i++){
        hist3d[i] = 0;
    }
    
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int bix = img.at<Vec3b>(i,j)[0]/divisor;
            int gix = img.at<Vec3b>(i,j)[1]/divisor;
            int rix = img.at<Vec3b>(i,j)[2]/divisor;
            hist3d[ rix*Hsize*Hsize + gix*Hsize + bix ]++;
        }
    }
    
    //sum of all bin values
    float numpx=0;
    for(int i=0;i<Hsize*Hsize*Hsize;i++){
        numpx+=hist3d[i];
    }
    
    //normalize values
    for(int i=0;i<Hsize*Hsize*Hsize;i++){
        hist3d[i] = hist3d[i]/numpx;
    }
    
    return 0;
}

//find 2d rg histogram of image
int hist2d(Mat &img, float **histd,int Hsize){
    
    for(int i=0;i<Hsize;i++){
        for(int j=0;j<Hsize;j++){
            histd[i][j]=0;
        }
    }
    
    
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int sum = img.at<Vec3b>(i,j)[0]+img.at<Vec3b>(i,j)[1]+img.at<Vec3b>(i,j)[2]+1;
            //c=1 Green
            int gix = (Hsize*img.at<Vec3b>(i,j)[1])/sum;
            //c=2 Red
            int rix =(Hsize*img.at<Vec3b>(i,j)[2])/sum;
            histd[rix][gix]++;
        }
    }
    
    float numpx = 0;
    //sum of all bin values
    for(int i=0;i<Hsize;i++){
        for(int j=0;j<Hsize;j++){
            numpx+=histd[i][j];
        }
    }
    
    //normalize values
    for(int i=0;i<Hsize;i++){
        for(int j=0;j<Hsize;j++){
            histd[i][j]=histd[i][j]/numpx;
        }
    }
    
    return 0;
}

//histogram intersection for 2d histogram
int hist_intersection(float **hist_targ, float **hist_db,vector<sort_data_type> &vec,string imgname,int Hsize){
    
    float s = 0;
    for(int i=0;i<Hsize;i++){
        for(int j=0;j<Hsize;j++){
            float f1 = hist_targ[i][j];
            float f2 = hist_db[i][j];
            if(f1<f2){
                s+=f1;
            }
            else{
                s+=f2;
            }
        }
    }
    float d = 1-s;
    sort_data_type obj;
    obj.img_name = imgname;
    obj.dist_metric = d;
    vec.push_back(obj);
    return 0;
}

//histogram interction for 3d and 1d histogram(named 3d but used for 1d as well)
int hist_intersection3d(float *hist_target, float *hist_db, vector<sort_data_type> &vec, string imgname, int size){

    float s = 0;
    for(int i=0;i<size;i++){
        float f1 = hist_target[i];
        float f2 = hist_db[i];
        
        if(f1<f2){
            s+=f1;
        }
        else{
            s+=f2;
        }
    }
    
    float d = 1-s;
    sort_data_type obj;
    obj.img_name = imgname;
    obj.dist_metric = d;
    vec.push_back(obj);
    return 0;
}

//crops centre 9x9 square from image
int square9x9(Mat &src, Mat &dst){
    //src is the image
    //dst is the 0x0 Mat containing all 0s
    int row = ((src.rows)/2)-4;
    int col = ((src.cols)/2)-4;
    int row_dst = 0;
    int col_dst = 0;
    
    int dims[2] = {9,9};
    dst=Mat::zeros(2, dims, CV_8UC3);
    
    for(int i=row;i<row+9;i++){
        col_dst=0;
        for(int j=col;j<col+9;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(row_dst,col_dst)[c] = src.at<Vec3b>(i,j)[c];
            }
            col_dst++;
        }
        row_dst++;
    }
    return 0;
};

//crops centre 300x300 square from image(for Task 5)
int square301x301(Mat &src, Mat &dst){
    //src is the image
    //dst is the 0x0 Mat containing all 0s
    int row = ((src.rows)/2)-150;
    int col = ((src.cols)/2)-150;
    int row_dst = 0;
    int col_dst = 0;
    
    int dims[2] = {301,301};
    dst=Mat::zeros(2, dims, CV_8UC3);
    
    for(int i=row;i<row+301;i++){
        col_dst=0;
        for(int j=col;j<col+301;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(row_dst,col_dst)[c] = src.at<Vec3b>(i,j)[c];
            }
            col_dst++;
        }
        row_dst++;
    }
    return 0;
};

//finds sum squared difference for 9x9 image
int sumsqdiff(Mat &target, Mat &dbimg){
    //cout << "in sum sq diff" << endl;
    int d=0;
    for(int i=0;i<9;i++){
        for(int j=0;j<9;j++){
            for(int c=0;c<3;c++){
                d+= ((target.at<Vec3b>(i,j)[c] - dbimg.at<Vec3b>(i,j)[c])*(target.at<Vec3b>(i,j)[c] - dbimg.at<Vec3b>(i,j)[c]))/81;
            }
        }
    }
    //cout << d << endl;
    return d;
};

//for quicksort
int Partition(vector<sort_data_type> &v, int start, int end){
    
    int pivot = end;
    int j = start;
    for(int i=start;i<end;++i){
        if(v[i].dist_metric<v[pivot].dist_metric){
            swap(v[i],v[j]);
            ++j;
        }
    }
    swap(v[j],v[pivot]);
    return j;
    
}

//quicksort
void Quicksort(vector<sort_data_type> &v, int start, int end ){
    
    if(start<end){
        int p = Partition(v,start,end);
        Quicksort(v,start,p-1);
        Quicksort(v,p+1,end);
    }
    
};

//find distance metric for task 1
int baseline_matching(Mat &src, Mat &temp, Mat &target_baseline, vector<sort_data_type> &dist,string img_name){
    int dims[2] = {9,9};
    temp=Mat::zeros(2, dims, CV_8UC3);
    
    //apply feature vector on db images
    square9x9(src, temp);
    
    //find distance_metric
    int dist_metr = sumsqdiff(target_baseline, temp);
    sort_data_type obj;
    obj.dist_metric = dist_metr;
    obj.img_name = img_name;
    dist.push_back(obj);
    return 0;
}

//prints top 3 matches
int printTop3(vector<sort_data_type> &vec){
    cout << "Top 3 matches: " << endl;
    for(int i=0;i<11;i++){
        cout << vec[i].img_name << ": " << vec[i].dist_metric << endl;
    }
    return 0;
}

//splits image horizontally into specified number of parts
int split_img(Mat &src, Mat &dst, int starting_row,int parts){
    int row_dst = 0;
    int col_dst = 0;
    
    int dims[2] = {src.rows/parts,src.cols};
    dst=Mat::zeros(2, dims, CV_8UC3);
    
    for(int i=starting_row;i<starting_row+(src.rows/parts);i++){
        col_dst=0;
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(row_dst,col_dst)[c] = src.at<Vec3b>(i,j)[c];
            }
            col_dst++;
        }
        row_dst++;
    }
    return 0;
}

//sobelx filter
int sobelX3x3( Mat &src, Mat3s &dst ){
    Mat3s tmp;
    src.copyTo(tmp);
    
    //[-1 0 1]
    for(int i=0;i<src.rows;i++){
        for(int j=1;j<src.cols-1;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c]=(src.at<Vec3b>(i,j-1)[c]*-1+src.at<Vec3b>(i,j+1)[c]);
            }
        }
    }
    
    //[1 2 1]T
    for(int i=1;i<src.rows-1;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3s>(i,j)[c]=(tmp.at<Vec3s>(i-1,j)[c]+tmp.at<Vec3s>(i,j)[c]*2+tmp.at<Vec3s>(i+1,j)[c])/4;
                
            }
        }
    }
    
    
    return 0;
};

//sobely filter
int sobelY3x3( Mat &src, Mat3s &dst ){
    Mat3s tmp;
    src.copyTo(tmp);
    //[1 2 1]
    for(int i=0;i<src.rows;i++){
        for(int j=1;j<src.cols-1;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3s>(i,j)[c]=(src.at<Vec3b>(i,j-1)[c]+src.at<Vec3b>(i,j)[c]*2+src.at<Vec3b>(i,j+1)[c])/4;
            }
        }
    }
    
    //[-1 0 1]T
    for(int i=1;i<src.rows-1;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3s>(i,j)[c]=(tmp.at<Vec3s>(i-1,j)[c]*-1+tmp.at<Vec3s>(i+1,j)[c]);
                
            }
        }
    }
    
    return 0;
};

//magintude from sobelx and sobely
int magnitude( Mat3s &sx, Mat3s &sy, Mat3s &dst ){
    for(int i=0;i<sx.rows;i++){
        for(int j=0;j<sx.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3s>(i,j)[c]=sqrt
                ((sx.at<Vec3s>(i,j)[c]*sx.at<Vec3s>(i,j)[c])
                 +(sy.at<Vec3s>(i,j)[c]*sy.at<Vec3s>(i,j)[c]));
            }
        }
    }
    return 0;
};

//feature_vector for Task 5
int feature_vector(Mat &target, float *hist1d_target, float *hist3d_targetf1, float *hist3d_targetf2, float *hist3d_targetf3, float *hist3d_targetf4, int Hsize){
    //target or training image
    Mat img,tmp;
    target.copyTo(img);

    cvtColor(img, tmp, COLOR_BGR2HSV);

    //1d histogram of only hue values
    for(int i=0;i<Hsize;i++){
        hist1d_target[i] = 0;
    }

    int divisor = 256/Hsize;

    for(int i=0;i<tmp.rows;i++){
        for(int j=0;j<tmp.cols;j++){
            int ix = (tmp.at<Vec3b>(i,j)[0])/divisor;
            hist1d_target[ix]++;
        }
    }
    //normalize 1d histogram
    float s = 0;
    for(int i=0;i<Hsize;i++){
        s+=hist1d_target[i];
    }
    
    for(int i=0;i<Hsize;i++){
        hist1d_target[i]/=s;
    }
    
    //laws filters on original image
    Mat t_f1,t_f2,t_f3,t_f4;
    laws_f1(target,t_f1);
    laws_f2(target,t_f2);
    laws_f3(target,t_f3);
    laws_f4(target,t_f4);
    hist3d(t_f1, hist3d_targetf1, Hsize);
    hist3d(t_f2, hist3d_targetf2, Hsize);
    hist3d(t_f3, hist3d_targetf3, Hsize);
    hist3d(t_f4, hist3d_targetf4, Hsize);
    return 0;
}

//Task1: baseline matching
int Task1(Mat &target, vector<sort_data_type> *vec){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    
    //9x9 of target image
    Mat target_baseline;
    square9x9(target, target_baseline);
    
    cout << "Enter directory name";
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
            
            Mat img,temp;
            
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            
            //call distance metric function
            baseline_matching(img, temp, target_baseline, *vec, dp->d_name);
        }
    }
    
    //processing
    Quicksort(*vec, 0, vec->size()-1);
    printTop3(*vec);
    vec->clear();
    return 0;
}

//Task2: Histogram Matching
int Task2(Mat &target, vector<sort_data_type> *vec){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    
    //2D histogram
    int Hsize;
    float **hist_target;
    float **hist_db;
    string dimensions;
    
    cout << "Enter dimensions rqd RGB/RG:(string) " << endl;
    cin >> dimensions;
    
    
    //3D histograms
    float *hist3d_target;
    float *hist3d_db;
    
    
    cout << "Enter number of bins" << endl;
    cin >> Hsize;
    if(dimensions == "RG" || dimensions == "rg"){
        //allocate space for target hist
        hist_target = new float*[Hsize];
        hist_target[0] = new float[Hsize*Hsize];
        for(int i=1;i<Hsize;i++){
            hist_target[i] = &(hist_target[0][i*Hsize]);
        }
        
        hist2d(target, hist_target,Hsize);
        
        //allocate space for db hist
        hist_db = new float*[Hsize];
        hist_db[0] = new float[Hsize*Hsize];
        for(int i=1;i<Hsize;i++){
            hist_db[i] = &(hist_db[0][i*Hsize]);
        }
    }
    else{
        hist3d_target = new float[Hsize*Hsize*Hsize];
        hist3d_db = new float[Hsize*Hsize*Hsize];
        hist3d(target, hist3d_target, Hsize);
    }
    
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
            
            if(dimensions == "RG"){
                //calculate feauture vector values
                hist2d(img, hist_db, Hsize);
                
                //calculate distance metric
                hist_intersection(hist_target, hist_db, *vec, dp->d_name, Hsize);
            }
            else{
                hist3d(img, hist3d_db, Hsize);
                hist_intersection3d(hist3d_target, hist3d_db, *vec, dp->d_name, Hsize*Hsize*Hsize);
            }
            
            
            
        }
    }
    
    //processing
    Quicksort(*vec, 0, vec->size()-1);
    printTop3(*vec);
    vec->clear();
    return 0;
}

//Task3: Multi Histogram Matching
int Task3(Mat &target, vector<sort_data_type> *vec, vector<sort_data_type> *arr){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int Hsize;
    int cs=0;
    
    cout << "Case 1: Histogram of whole image, 9x9 square in middle" << endl;
    cout << "Case 2: Histogram of top half and bottom half" << endl;
    cout << "Enter case number 1/2?: " << endl;
    cin >> cs;
    
    cout << "Enter number of bins" << endl;
    cin >> Hsize;
    
    float *hist3d_target = new float[Hsize*Hsize*Hsize];
    float *hist3d_db = new float[Hsize*Hsize*Hsize];
    float *hist3d_secdb = new float[Hsize*Hsize*Hsize];
    float *hist3d_secTarget = new float[Hsize*Hsize*Hsize];
    Mat target_baseline,top_half,bottom_half;
    
    //target image
    if(cs == 1){
        //case1: assuming same #bins for both histograms
        
        
        //full image, middle of image
        hist3d(target, hist3d_target, Hsize);
        //find 3D histogram for 9x9 of target image
        square9x9(target, target_baseline);
        hist3d(target_baseline, hist3d_secTarget, Hsize);
    }
    else{
        split_img(target, top_half, 0,2);
        hist3d(top_half, hist3d_target, Hsize);

        split_img(target,bottom_half,target.rows/2,2);
        hist3d(bottom_half,hist3d_secTarget,Hsize);
    }
    
    //directory
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
            
            Mat img,temp,middle_db,t_half,b_half;
            
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            
            //comparision image histograms
            if(cs == 1){
                //full image, middle of image
                
                //entire image hist
                hist3d(img,hist3d_db, Hsize);
                
                //9x9 in middle hist
                square9x9(img, middle_db);
                hist3d(middle_db, hist3d_secdb, Hsize);
            }
            else{
                //top half, bottom half
                split_img(img, t_half, 0,2);
                hist3d(t_half,hist3d_db,Hsize);

                split_img(img,b_half,img.rows/2,2);
                hist3d(b_half,hist3d_secdb,Hsize);
            }
            
            hist_intersection3d(hist3d_target, hist3d_db, *vec, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_secTarget, hist3d_secdb, *arr, dp->d_name, Hsize*Hsize*Hsize);
            
        }
    }

    for(int i=0;i<vec->size();i++){
        float midval = (0.5*vec->at(i).dist_metric) + (0.5*arr->at(i).dist_metric);
        vec->at(i).dist_metric = midval;
    }
    
    Quicksort(*vec, 0, vec->size()-1);
    printTop3(*vec);
    vec->clear();
    arr->clear();
    return 0;
}

//Task4: Texture and Color
int Task4(Mat &target,vector<sort_data_type> *vec, vector<sort_data_type> *arr){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int Hsize;
    cout << "Enter number of bins" << endl;
    cin >> Hsize;
    float *hist3d_target = new float[Hsize*Hsize*Hsize];
    float *hist3d_db = new float[Hsize*Hsize*Hsize];
    float *hist3d_secdb = new float[Hsize*Hsize*Hsize];
    float *hist3d_secTarget = new float[Hsize*Hsize*Hsize];
    
    //target image
    Mat3s sobx,soby,mag;
    Mat dst;
    target.copyTo(sobx);
    target.copyTo(soby);
    target.copyTo(mag);
    sobelX3x3(target, sobx);
    sobelY3x3(target, soby);
    magnitude(sobx, soby, mag);
    convertScaleAbs(mag, dst);

    hist3d(target,hist3d_target,Hsize);
    hist3d(dst, hist3d_secTarget, Hsize);
    
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
            
            Mat img,temp;
            Mat3s sob_x,sob_y,magn;
            
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            img.copyTo(sob_x);
            img.copyTo(sob_y);
            img.copyTo(magn);
            img.copyTo(temp);
            
            sobelX3x3(img, sob_x);
            sobelY3x3(img, sob_y);
            magnitude(sob_x, sob_y, magn);
            convertScaleAbs(magn, temp);

            hist3d(img,hist3d_db,Hsize);
            hist3d(temp,hist3d_secdb,Hsize);
            hist_intersection3d(hist3d_target, hist3d_db, *vec, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_secTarget, hist3d_secdb, *arr, dp->d_name, Hsize*Hsize*Hsize);
        }
    }
    
    for(int i=0;i<vec->size();i++){
        float midval = (0.5*vec->at(i).dist_metric) + (0.5*arr->at(i).dist_metric);
        vec->at(i).dist_metric = midval;
    }
    
    Quicksort(*vec, 0, vec->size()-1);
    printTop3(*vec);
    vec->clear();
    arr->clear();
    return 0;
}

//Task5: Custom design
int Task5(Mat &target, vector<sort_data_type> *vec1, vector<sort_data_type> *vec2, vector<sort_data_type> *vec3, vector<sort_data_type> *vec4, vector<sort_data_type> *vec5){
    
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int Hsize;
    cout << "Enter number of bins for 3D histogram" << endl;
    cin >> Hsize;
    
    float *hist1d_target = new float[Hsize];
    float *hist3d_targetf1 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf2 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf3 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf4 = new float[Hsize*Hsize*Hsize];
    
    float *hist1d_db = new float[Hsize];
    float *hist3d_dbf1 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf2 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf3 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf4 = new float[Hsize*Hsize*Hsize];
    
    feature_vector(target, hist1d_target,hist3d_targetf1,hist3d_targetf2,hist3d_targetf3,hist3d_targetf4,Hsize);
    
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
        
            Mat db,img;
            
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            db = imread(buffer, IMREAD_COLOR);
            if(db.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            //focus only on middle 300x300 square
            square301x301(db, img);
            feature_vector(img, hist1d_db,hist3d_dbf1,hist3d_dbf2,hist3d_dbf3,hist3d_dbf4,Hsize);
   
            //Histogram Intersection
            hist_intersection3d(hist1d_target, hist1d_db, *vec1, dp->d_name, Hsize);
            hist_intersection3d(hist3d_targetf1, hist3d_dbf1, *vec2, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf2, hist3d_dbf2, *vec3, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf3, hist3d_dbf3, *vec4, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf4, hist3d_dbf4, *vec5, dp->d_name, Hsize*Hsize*Hsize);
                       
        }
    }
    for(int i=0;i<vec1->size();i++){
        float midval = (0.6*vec1->at(i).dist_metric) + (0.1*vec2->at(i).dist_metric)+(0.1*vec3->at(i).dist_metric)+(0.1*vec4->at(i).dist_metric)+(0.1*vec5->at(i).dist_metric);
        vec1->at(i).dist_metric = midval;
    }
    Quicksort(*vec1, 0, vec1->size()-1);
    printTop3(*vec1);
    vec1->clear();
    vec2->clear();
    vec3->clear();
    vec4->clear();
    vec5->clear();
    return 0;
}

/*------------------------------------------------------------------------------------------------------------------
                                                EXTENSIONS
--------------------------------------------------------------------------------------------------------------------*/

//------------------------------- Gabor -------------------------------------

int gaborfilter(Mat &src, float *hist1d, int bins){
    Mat temp;
    cvtColor(src,temp,COLOR_BGR2GRAY);
    int ker_size = 9;
    double sig = 5, lmd = 4, gamma = 0.04, psi = CV_PI/4;
    double theta[] = {0,CV_PI/12,CV_PI/4,CV_PI/2-CV_PI/12,CV_PI/2,CV_PI/12-CV_PI/2,-CV_PI/4,-CV_PI/12};
    int arrSize = sizeof(theta)/sizeof(theta[0]);
    
    for(int i=0;i<bins*arrSize;i++){
        hist1d[i]=0;
    }
    int divisor = 256/bins;
    
    for(int i=0;i<arrSize;i++){
        Mat tmp;
        Mat filter = getGaborKernel(Size(ker_size,ker_size), sig, theta[i],lmd,gamma,psi,CV_32F);

        filter2D(src, tmp, CV_32F, filter);
//        imshow("gabor",tmp);
//        waitKey(0);
        //1d histogram of tmp
        for(int j=0;j<tmp.rows;j++){
            for(int k=0;k<tmp.cols;k++){
                int ix = tmp.at<Vec3b>(j,k)[0]/divisor;
                hist1d[(i*bins)+ix]++;
            }
        }
    }
    
    //    normalize
    float s=0;
    for(int l=0;l<bins*arrSize;l++){
        s+=hist1d[l];
    }
    for(int l=0;l<bins*arrSize;l++){
        hist1d[l]/=s;
    }
    
    return 0;
}

float sum_squared(float *hist1d_target, float *hist1d_db, int size){
    float d=0;
    for(int i=0;i<size;i++){
        d+=((hist1d_target[i]-hist1d_db[i])*(hist1d_target[i]-hist1d_db[i]))/size;
    }
    return d;
}

int gab_texture(Mat &target, vector<sort_data_type> *vec, vector<sort_data_type> *arr){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int Hsize;
    int bins;
    cout << "Enter number of bins for 3D histogram" << endl;
    cin >> Hsize;
    float dist_metric_3d = 0;
    float dist_metric_1d = 0;
    float dist_metr = 0;
    //entire image histogram
    float *hist3d_target = new float[Hsize*Hsize*Hsize];
    float *hist3d_db = new float[Hsize*Hsize*Hsize];
    
    cout << "Enter number of bins for 1D histogram" << endl;
    cin >> bins;
    
    string dm_flag = "";
    cout << "HI OR SSD? " << endl;
    cin >> dm_flag;
    //gabor filter histograms
    float *hist1d_target = new float[bins*8];
    float *hist1d_db = new float[bins*8];
    
    //target image
    hist3d(target, hist3d_target, Hsize);
    gaborfilter(target, hist1d_target, bins);
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
            
            hist3d(img,hist3d_db,Hsize);
            gaborfilter(img, hist1d_db, bins);
            
            
            if(dm_flag == "HI"){
                //histogram intersection
                hist_intersection3d(hist3d_target, hist3d_db, *vec, dp->d_name, Hsize*Hsize*Hsize);
                hist_intersection3d(hist1d_target, hist1d_db, *arr, dp->d_name, bins*8);
            }
            else{
                //            sum squared difference
                //             distance metric for entire image
                dist_metric_3d = sum_squared(hist3d_target, hist3d_db, Hsize*Hsize*Hsize);
                //distance metric for gabor filters
                dist_metric_1d = sum_squared(hist1d_target, hist1d_db, bins*8);
                dist_metr = (0.7*dist_metric_1d)+(0.3*dist_metric_3d);
                sort_data_type obj;
                obj.img_name = dp->d_name;
                obj.dist_metric = dist_metr;
                vec->push_back(obj);
            }
        }
    }
    
    if(dm_flag == "HI"){
        for(int i=0;i<vec->size();i++){
            float midval = (0.3*vec->at(i).dist_metric) + (0.7*arr->at(i).dist_metric);
            vec->at(i).dist_metric = midval;
        }
    }
    
    Quicksort(*vec, 0, vec->size()-1);
    printTop3(*vec);
    vec->clear();
    arr->clear();
    return 0;
}

//----------------------------------Laws--------------------------------------


int laws_fil(Mat &target, vector<sort_data_type> *vec1, vector<sort_data_type> *vec2, vector<sort_data_type> *vec3, vector<sort_data_type> *vec4, vector<sort_data_type> *vec5){
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int Hsize;
    cout << "Enter number of bins for 3D histogram" << endl;
    cin >> Hsize;
    Mat t_f1,t_f2,t_f3,t_f4,tmp;
    
    float *hist3d_target = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf1 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf2 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf3 = new float[Hsize*Hsize*Hsize];
    float *hist3d_targetf4 = new float[Hsize*Hsize*Hsize];
    
    float *hist3d_db = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf1 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf2 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf3 = new float[Hsize*Hsize*Hsize];
    float *hist3d_dbf4 = new float[Hsize*Hsize*Hsize];
    
    hist3d(target, hist3d_target, Hsize);
    laws_f1(target,t_f1);
    laws_f2(target,t_f2);
    laws_f3(target,t_f3);
    laws_f4(target,t_f4);
    hist3d(t_f1, hist3d_targetf1, Hsize);
    hist3d(t_f2, hist3d_targetf2, Hsize);
    hist3d(t_f3, hist3d_targetf3, Hsize);
    hist3d(t_f4, hist3d_targetf4, Hsize);
    
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
            
            Mat img,db_f1,db_f2,db_f3,db_f4,temp;
            
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            img = imread(buffer, IMREAD_COLOR);
            if(img.empty())
            {
                cout << "Could not access the image" << endl;
                return 1;
            }
            
            hist3d(img,hist3d_db,Hsize);
            laws_f1(img,db_f1);
            laws_f2(img,db_f2);
            laws_f3(img,db_f3);
            laws_f4(img,db_f4);
            hist3d(db_f1, hist3d_dbf1, Hsize);
            hist3d(db_f2, hist3d_dbf2, Hsize);
            hist3d(db_f3, hist3d_dbf3, Hsize);
            hist3d(db_f4, hist3d_dbf4, Hsize);
            
            hist_intersection3d(hist3d_target, hist3d_db, *vec1, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf1, hist3d_dbf1, *vec2, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf2, hist3d_dbf2, *vec3, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf3, hist3d_dbf3, *vec4, dp->d_name, Hsize*Hsize*Hsize);
            hist_intersection3d(hist3d_targetf4, hist3d_dbf4, *vec5, dp->d_name, Hsize*Hsize*Hsize);
            
           
        }
    }
    
    for(int i=0;i<vec1->size();i++){
        float midval = (0.2*vec1->at(i).dist_metric) + (0.2*vec2->at(i).dist_metric)+(0.2*vec3->at(i).dist_metric)+(0.2*vec4->at(i).dist_metric)+(0.2*vec5->at(i).dist_metric);
        vec1->at(i).dist_metric = midval;
    }
    Quicksort(*vec1, 0, vec1->size()-1);
    printTop3(*vec1);
    vec1->clear();
    vec2->clear();
    vec3->clear();
    vec4->clear();
    vec5->clear();
    return 0;
}




#endif /* tasks_hpp */
