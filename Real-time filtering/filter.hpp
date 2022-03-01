//
//  filter.hpp
//  CV
//
//  Created by Sumegha Singhania on 1/27/22.
//

#ifndef filter_hpp
#define filter_hpp


#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int alt_greyscale(Mat &src, Mat &dst){
    
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<2;c++){
                dst.at<Vec3b>(i,j)[c]=src.at<Vec3b>(i,j)[2];
            }
            
        }
    }
    return 0;
};

int blur5x5( Mat &src, Mat &dst ){
    Mat tmp;
    src.copyTo(tmp);
   //for(int i=0;i<10;i++){
    for(int i=0;i<src.rows;i++){
        for(int j=2;j<src.cols-2;j++){
            for(int c=0;c<3;c++){
                tmp.at<Vec3b>(i,j)[c] = (src.at<Vec3b>(i,j-2)[c]+src.at<Vec3b>(i,j-1)[c]*2+src.at<Vec3b>(i,j)[c]*4+src.at<Vec3b>(i,j+1)[c]*2+src.at<Vec3b>(i,j+2)[c])/10;
                
            }
        }
    }
    for(int i=2;i<src.rows-2;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(i,j)[c] = (tmp.at<Vec3b>(i-2,j)[c]+tmp.at<Vec3b>(i-1,j)[c]*2+tmp.at<Vec3b>(i,j)[c]*4+tmp.at<Vec3b>(i+1,j)[c]*2+tmp.at<Vec3b>(i+2,j)[c])/10;
                
            }
        }
    }
//}
    
    return 0;
};

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

int blurQuantize( Mat &src, Mat &dst, int levels ){
    blur5x5(src, dst);
    int b = 255/levels;
    for(int i=0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            for(int c=0;c<3;c++){
                int val = dst.at<Vec3b>(i,j)[c]/b;
                val=val*b;
                dst.at<Vec3b>(i,j)[c]=val;
            }
        }
    }
    
    return 0;
};

int cartoon( Mat &src, Mat&dst, int levels, int magThreshold ){
    Mat3s sob_x,sob_y,mag,tmp;
    src.copyTo(sob_x);
    src.copyTo(sob_y);
    src.copyTo(mag);
    src.copyTo(tmp);
    sobelX3x3(src, sob_x);
    sobelY3x3(src, sob_y);
    magnitude(sob_x, sob_y, mag);
    blurQuantize(src, dst, levels);
    for(int i=0;i<mag.rows;i++){
        for(int j=0;j<mag.cols;j++){
            for(int c=0;c<3;c++){
                if(mag.at<Vec3s>(i,j)[c]>magThreshold){
                    dst.at<Vec3b>(i,j)[c]=0;
                }
            }
        }
    }
    return 0;
};

int negative(Mat &src, Mat &dst){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(i,j)[c]=255 - src.at<Vec3b>(i,j)[c];
            }
        }
    }
    return 0;
};

int flip(Mat &src, Mat &dst){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(i,j)[c]=src.at<Vec3b>(i,src.cols-j-1)[c];
            }
        }
    }
    return 0;
};

int split(Mat &src,Mat &dst){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            for(int c=0;c<3;c++){
                dst.at<Vec3b>(i,j)[c]=src.at<Vec3b>(i,src.cols/2-j)[c];
            }
        }
    };
    return 0;
}

int rbg_filter(Mat &src, Mat &dst,string str){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            if(str == "red"){
                for(int c=0;c<2;c++){
                dst.at<Vec3b>(i,j)[c]=0;
                }}
            else if(str == "green"){
                for(int c=0;c<3&&c!=1;c++){
                    dst.at<Vec3b>(i,j)[c]=0;
                    }
                }
            else if(str == "blue"){
                for(int c=1;c<3;c++){
                    dst.at<Vec3b>(i,j)[c]=0;
                }
            }
            
        }
    }
    return 0;
};

int sketch(Mat &src,Mat &dst){
    Mat3s sob_x,sob_y,mag;
    Mat tmp1,tmp2;
    
    src.copyTo(tmp1);
    src.copyTo(tmp2);
    src.copyTo(sob_x);
    src.copyTo(sob_y);
    src.copyTo(mag);
    src.copyTo(dst);
    sobelX3x3(src, sob_x);
    sobelY3x3(src, sob_y);
    magnitude(sob_x, sob_y, mag);
    convertScaleAbs(mag, tmp1);
    negative(tmp1, tmp2);
    cvtColor(tmp2, dst, COLOR_RGB2GRAY);
    return 0;
}

int color_pop(Mat &src, Mat &dst){
    Mat hsv,gray,mask,mask_inv,res,backg,mask2;
    src.copyTo(res);
    src.copyTo(gray);
    src.copyTo(hsv);
    src.copyTo(mask);
    src.copyTo(mask_inv);
    src.copyTo(backg);
    cvtColor(src, hsv, COLOR_BGR2HSV);
    cvtColor(src, gray, COLOR_BGR2GRAY);
    cvtColor(gray, gray, COLOR_GRAY2BGR);
    inRange(hsv, Scalar(160, 100, 50), Scalar(180, 255, 255), mask);
    inRange(hsv, Scalar(0, 100, 50), Scalar(10, 255, 255), mask2);
    bitwise_or(mask, mask2, mask);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    bitwise_not(mask, mask_inv);
    bitwise_and(src, mask, res);
    bitwise_and(gray, mask_inv, backg);
    add(res, backg, dst);
   
    return 0;
}
#endif /* filter_hpp */
