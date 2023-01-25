/*
 Tasks.h:
 Contains functions corresponding to the core tasks and extensions.
 The tasks are divided as follows:
 
 //Core Tasks
 1. Function: calibrate_camera(): Tasks 1-3
    a. Keypress 's': saves the calibration image, its corresponding world and image points
    b. Keypress 'c':
 2. Function: projection(): Tasks 4-6
 3. Function: draw_shape(): Task 6
 
 //Extensions
 1. Project on multiple targets in a scene (included in projection())
 2. Insert object on static images and pre recorded videos (included in projection())
 3. Change target to something else (included in draw_shape())
 4. Project video on target: alter_target()
 5. Tested out with different cameras
 
 Further details of functions are given above each of them.
 */
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "csv_util.h"
using namespace cv;
using namespace std;

#ifndef calibration_h
#define calibration_h

//defining the dimensions of the chessboard
int chessboard[2]{9,6};

//Contains all functions corresponding to calibrating the camera (Task 1-3)
int calibrate_camera(){
    
//------------------------------------------- define variables---------------------------------------------------------------
    Mat frame;
    int count=1;
    vector<Point2f> corners; //2d img points
    vector<Vec3f> points; //3d world points constructed by us
    vector<vector<Vec3f> > points_list; //vector to store 3d world points
    vector<vector<Point2f> > corners_list; //vector to store 2d img points
    
    
    //initializing 3d world points vector
    //initialized outside because they will be the same for all saved frames
    for(int i=0;i<chessboard[1];i++){
        for(int j=0;j<chessboard[0];j++){
            points.push_back(Vec3f(j,-1*i,0));
        }
    }
    
//---------------------------------- Capture calibration frames from video -------------------------------------------------------
    //Video capture
    VideoCapture *capdev;
    
    // open the video device (0 for webcam, 1 for phone camera, Extension 5)
    capdev = new VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }
    
    namedWindow("Video", 1);
    for(;;) {
        
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        
        //To save the captures frame and it's corners
        Mat gray,save_frame;
        vector<Point2f> last_corners;
        
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        //function to findchessboardcorners in the frame
        int found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        //Task1: if we find a chessboard, we save the frame, corners and draw corners of the chessboard
        if(found){
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
            
            //    refining pixel coordinates for given 2d points.
            cornerSubPix(gray,corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
            
            gray.copyTo(save_frame);
            for(int i=0;i<corners.size();i++){
                last_corners.push_back(corners[i]);
            }
            // Displaying the detected corner points on the checker board
            cv::drawChessboardCorners(frame, Size(chessboard[0], chessboard[1]), corners, found);
        }
        
        
        imshow("Video", frame);
        
        // see if there is a waiting keystroke
        char key = cv::waitKey(10);
        
        //On keypress 's' saves the calibration image and its points and corners (Task2)
        if (key == 's'){
            string str = "/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/Calib_frames/frame"+to_string(count)+".jpg";
            imwrite(str, save_frame);
            count++;
            cout << "Number of corners: " << last_corners.size() << endl;
            cout << "First corner: " << last_corners[0].x << ", " << last_corners[0].y << endl;
            points_list.push_back(points);
            corners_list.push_back(vector<Point2f> (last_corners));
        }
        
        //On keypress 'c', calibrates the cameras and finds the corresponding matrices (Task 3)
        if(key == 'c'){
            
            //calibrates only if there are atleast 5 images and their data saved
            if(count>=5){
                
                cout << "Starting Camera Calibration: " << endl;
                Mat camera_mat, dist_coeff, R,T;
                
                //calibrate camera and find reprojection error
                double error = calibrateCamera(points_list, corners_list, Size(frame.rows,frame.cols), camera_mat, dist_coeff, R, T,CALIB_FIX_ASPECT_RATIO);
                
                cout << "re-projection error: " << error << endl;
                cout << "camera matrix: " << camera_mat << endl;
                cout << "distance coeff: " << dist_coeff << endl;
                cout << "Rotation vector: " << R << endl;
                cout << "Translation vector: " << T << endl;
                
                //to store the camera matrix and distance coeff matrix in a csv file
                vector<double> data;
                for(int i=0;i<camera_mat.rows;i++){
                    for(int j=0;j<camera_mat.cols;j++){
                        data.push_back(camera_mat.at<double_t>(i,j));
                    }
                }
                for(int i=0;i<dist_coeff.rows;i++){
                    for(int j=0;j<dist_coeff.cols;j++){
                        data.push_back(dist_coeff.at<double_t>(i,j));
                    }
                }
                
                //writing data to a csv file
                append_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/calib_data.csv","", data, 1);
            }
            else{
                
                //ask to capture more images if less than 5
                cout << "Capture more calibration frames" << endl;
            }
        }
        
        //if keypress 'q' break the video loop
        if( key == 'q') {
            break;
        }
    }
    delete capdev;
    return 0;
}

//Inserts video into target chessboard area
int alter_target(Mat &frame, Mat &R, Mat &T, Mat &cam_mat, Mat &dist_coeff,vector<Point2f> &corners ){
    
    //image that will be altered
    Mat markerImage;
    
    //open the video
    string vid_path;
    cout << "Enter video path: " << endl;
    cin >> vid_path;
    VideoCapture vid_capture(vid_path);
    if (!vid_capture.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
    }
    else
    {
        //Obtain fps and frame count by get() method and print
        int fps = vid_capture.get(5);
        cout << "Frames per second :" << fps;
        int frame_count = vid_capture.get(7);
        cout << "Frame count :" << frame_count;
    }
    while (vid_capture.isOpened())
    {
        // Initialize frame matrix
        Mat im_src;
        // Initialize a boolean to check if frames are there or not
        bool isSuccess = vid_capture.read(im_src);
        // If frames are present, show it
        if(isSuccess == true)
        {
            //corners of target chessboard
            vector<Point> pts_dst;
            pts_dst.push_back(corners[0]);
            pts_dst.push_back(corners[8]);
            pts_dst.push_back(corners[53]);
            pts_dst.push_back(corners[45]);
            
            //corners of replacement video frame
            vector<Point> pts_src;
            pts_src.push_back(Point(0,0));
            pts_src.push_back(Point(im_src.cols, 0));
            pts_src.push_back(Point(im_src.cols, im_src.rows));
            pts_src.push_back(Point(0, im_src.rows));
            
            // Compute homography from source and destination points
            Mat h = cv::findHomography(pts_src, pts_dst);
            
            // Warped image
            Mat warpedImage;
            // Warp source image to destination based on homography
            warpPerspective(im_src, warpedImage, h, frame.size(), INTER_CUBIC);
            
            // Prepare a mask representing region to copy from the warped image into the original frame.
            Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255), LINE_AA);
            
            // Erode the mask to not copy the boundary effects from the warping
            Mat element = getStructuringElement( MORPH_RECT, Size(5,5));
            //            Mat element = getStructuringElement( MORPH_RECT, Size(3,3));
            erode(mask, mask, element);
            
            // Copy the warped image into the original frame in the mask region.
            Mat imOut = frame.clone();
            warpedImage.copyTo(imOut, mask);
            imOut.copyTo(frame);
            
            //display frames
            imshow("Frame", frame);
        }
        
        // If frames are not there, close it
        if (isSuccess == false)
        {
            cout << "Video camera is disconnected" << endl;
            break;
        }
        //wait 20 ms between successive frames and break the loop if key q is pressed
        int key = waitKey(20);
        if (key == 'q')
        {
            cout << "q key is pressed by the user. Stopping the video" << endl;
            break;
        }
    }
    
    
    return 0;
    
}

//inserts virtual object in frame
int draw_shape(Mat &frame, Mat &R, Mat &T, Mat &cam_mat, Mat &dist_coeff,vector<Point2f> corners){
    Mat markerImage;
    
//---------------------------------------- Change target to something else (Extension 3)-------------------------------------------
    
    //image to replace target with
    Mat im_src = imread("/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/test3.jpeg",IMREAD_COLOR);

    //pushing 4 corneres of target chessboard to vector
    vector<Point> pts_dst;
    pts_dst.push_back(corners[0]);
    pts_dst.push_back(corners[8]);
    pts_dst.push_back(corners[53]);
    pts_dst.push_back(corners[45]);
    
    //pushing 4 croners of replacement image to vector
    vector<Point> pts_src;
    pts_src.push_back(Point(0,0));
    pts_src.push_back(Point(im_src.cols, 0));
    pts_src.push_back(Point(im_src.cols, im_src.rows));
    pts_src.push_back(Point(0, im_src.rows));
    
    
    // Compute homography from source and destination points
    Mat h = cv::findHomography(pts_src, pts_dst);
    
    // Warped image
    Mat warpedImage;
    // Warp source image to destination based on homography to fit it according to camera pose
    warpPerspective(im_src, warpedImage, h, frame.size(), INTER_CUBIC);
    
    // Prepare a mask representing region to copy from the warped image into the original frame.
    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255), LINE_AA);

    // Erode the mask to not copy the boundary effects from the warping
    Mat element = getStructuringElement( MORPH_RECT, Size(5,5));
    erode(mask, mask, element);

    // Copy the warped image into the original frame in the mask region.
    Mat imOut = frame.clone();
    warpedImage.copyTo(imOut, mask);
    imOut.copyTo(frame);
    
    
//------------------------------------------------- Insert virtual object (Task 6) --------------------------------------------------
    
    vector<Point2f> outp;
    
    //push points that require projecting
    vector<Vec3f> points;
    Vec3f a = {0,0,3};
    Vec3f b = {3,0,8};
    Vec3f c = {8,0,3};
    Vec3f d = {0,-5,3};
    Vec3f e = {5,-5,8};
    Vec3f f = {8,-5,3};
    points.push_back(a);
    points.push_back(b);
    points.push_back(c);
    points.push_back(d);
    points.push_back(e);
    points.push_back(f);
    
    //project 3D points
    projectPoints(points, R, T, cam_mat, dist_coeff, outp);
    
    //draw the shape
    line(frame, corners[0], corners[8], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[0], corners[45], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[8], corners[53], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[45], corners[53], Scalar(255,0,0),3,LINE_8);
    
    line(frame, corners[0], outp[0], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[8], outp[2], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[45], outp[3], Scalar(255,0,0),3,LINE_8);
    line(frame, corners[53], outp[5], Scalar(255,0,0),3,LINE_8);
    
    line(frame, outp[0], outp[1], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[0], outp[3], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[1], outp[2], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[1], outp[4], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[2], outp[5], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[3], outp[4], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[4], outp[5], Scalar(255,0,0),3,LINE_8);
    
    line(frame, outp[0], outp[3], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[0], outp[2], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[2], outp[5], Scalar(255,0,0),3,LINE_8);
    line(frame, outp[3], outp[5], Scalar(255,0,0),3,LINE_8);
    
    return 0;
}

//Contains all functions corresponding to projecting a virtual object into 3d space
int projection(){
    
    
//--------------------------------------------------- Read matrices from csv file-------------------------------------------------------------------
    //vectors to read data from the csv file
    vector<char *> names;
    vector<vector<double> > data;

    
    read_image_data_csv("/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/calib_data.csv", names, data, 0);

    Mat cam_mat;
    int dims[2] = {3,3};
    cam_mat = Mat::zeros(2, dims, CV_64FC1);

    int c=0;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            cam_mat.at<double_t>(i,j) = data[0][c];
            c++;
        }
    }

    Mat dist_coeff;
    int dim[2] = {5,1};
    dist_coeff = Mat::zeros(2, dim, CV_64FC1);
    c=9;
    for(int i=0;i<5;i++){
        dist_coeff.at<double_t>(0,i) = data[0][c];
        c++;
    }

//---------------------------------------------- defining variables-----------------------------------------------------------------------------
    //count for saving frame
    int count = 1;
    Mat frame;
    vector<Point2f> corners; //2d img points
    vector<Vec3f> points; //3d world points constructed by us
    vector<vector<Vec3f> > points_list; //vector to store 3d world points
    vector<vector<Point2f> > corners_list; //vector to store 2d img points
    
    //initializing 3d world points vector
    for(int i=0;i<chessboard[1];i++){
        for(int j=0;j<chessboard[0];j++){
            points.push_back(Vec3f(j,-1*i,1));
        }
    }
    
    //flag for the required application of virtual object insertion
    int flag = 0;
    cout << "For projection on static image: 0" << endl;
    cout << "For projection on recorded video: 1" << endl;
    cout << "For projection on live video: 2" << endl;
    cin >> flag;
    
    //virtual object insertion on static image
    if (flag == 0){
        
        string img_path;
        cout << "Enter image path: " << endl;
        cin >> img_path;
        Mat frame = imread(img_path, IMREAD_COLOR);
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        //finding chessboard corners on target image
        int found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        //loops over all possible chessboard corners found in the image
        while(found){
            
            //finds the 3d world points, image points corresponding to target chessboard corners
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
            //    refining pixel coordinates for given 2d points.
            cornerSubPix(gray,corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
            drawChessboardCorners(frame, Size(chessboard[0], chessboard[1]), corners, found);
            points_list.push_back(points);
            corners_list.push_back(vector<Point2f> (corners));
            
            //calculate current camera position (Task 4)
            Mat R,T;
            solvePnP(points, corners, cam_mat, dist_coeff, R, T);
            
            //*********Task5: done only on live video*********
            
            //inserts video in the target chessboard area (Extension 4)
            alter_target(frame, R, T, cam_mat, dist_coeff, corners);
            
            //inserts virtual object (Task 6)
            //also changes the target chessboard (Extension 3)
            draw_shape(frame, R, T, cam_mat, dist_coeff,corners);
            
            
            // to enable insertion of virtual object on multiple targets in the same scene
            found = 0;
            //remove this chessboard(turn all the pixels of the chessboard to white) and check for another chessboard
            for(int i=corners[0].y;i<=corners[53].y;i++){
                for(int j=corners[0].x;j<=corners[53].x;j++){
                    gray.at<uchar>(i,j) = 255;
                }
            }
            
            //check for chessboard again
            found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        }
        
        imshow("Image", frame);
        
    }
    
    //virtual object insertion on pre precorded video
    else if(flag == 1){
        
        //enter pre-recorded video
        string vid_path;
        cout << "Enter video path: " << endl;
        cin >> vid_path;
        VideoCapture vid_capture(vid_path);
        if (!vid_capture.isOpened())
            {
                cout << "Error opening video stream or file" << endl;
            }
        else
            {
                    // Obtain fps and frame count by get() method and print
                int fps = vid_capture.get(5);
                cout << "Frames per second :" << fps;
                int frame_count = vid_capture.get(7);
                cout << "Frame count :" << frame_count;
            }
        
        //read the video frame by frame
        while (vid_capture.isOpened())
        {
                // Initialize frame matrix
                Mat frame;
                // Initialize a boolean to check if frames are there or not
                bool isSuccess = vid_capture.read(frame);
                // If frames are present, show it
                if(isSuccess == true)
                {
                    Mat gray;
                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                    
                    //finding chessboard corners on target image
                    int found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    
                    //loops over all possible chessboard corners found in the image
                    while(found){
                        
                        //finds the 3d world points, image points corresponding to target chessboard corners
                        TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
                        //    refining pixel coordinates for given 2d points.
                        cornerSubPix(gray,corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
                        drawChessboardCorners(frame, Size(chessboard[0], chessboard[1]), corners, found);
                        points_list.push_back(points);
                        corners_list.push_back(vector<Point2f> (corners));
                        
                        //calculate current camera position (Task 4)
                        Mat R,T;
                        solvePnP(points, corners, cam_mat, dist_coeff, R, T);
                        
                        //inserts virtual object (Task 6)
                        //also changes the target chessboard (Extension 3)
                        draw_shape(frame, R, T, cam_mat, dist_coeff,corners);
                        
                        // to enable insertion of virtual object on multiple targets in the same scene
                        found = 0;
                        //remove this chessboard(turn all the pixels of the chessboard to white) and check for another chessboard
                        for(int i=corners[0].y;i<=corners[53].y;i++){
                            for(int j=corners[0].x;j<=corners[53].x;j++){
                                gray.at<uchar>(i,j) = 255;
                            }
                        }
                        
                        //check for chessboard again
                        found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                    }
                    
                    //display frames
                    imshow("Frame", frame);
                }

                // If frames are not there, close it
                if (isSuccess == false)
                {
                    cout << "Video camera is disconnected" << endl;
                    break;
                }
        //wait 20 ms between successive frames and break the loop if key q is pressed
                int key = waitKey(20);
                    if (key == 'q')
                {
                    cout << "q key is pressed by the user. Stopping the video" << endl;
                    break;
                }
            }
        // Release video capture object
        vid_capture.release();
        destroyAllWindows();
    }
    
    //virtual object insertion on live video
    else if (flag == 2){

        
        VideoCapture *capdev;

        // open the video device
        capdev = new VideoCapture(0);
        if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
        }

        namedWindow("Video", 1);
        for(;;) {

            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if( frame.empty() ) {
                printf("frame is empty\n");
                break;
            }
            
            
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            
            //finding chessboard corners on target frame
            int found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

            //loops over all possible chessboard corners found in the image
            while(found){
                
                //finds the 3d world points, image points corresponding to target chessboard corners
                TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
                //    refining pixel coordinates for given 2d points.
                cornerSubPix(gray,corners,cv::Size(11,11), cv::Size(-1,-1),criteria);
                cv::drawChessboardCorners(frame, Size(chessboard[0], chessboard[1]), corners, found);
                points_list.push_back(points);
                corners_list.push_back(vector<Point2f> (corners));
                
                //calculate current camera position (Task 4)
                Mat R,T;
                solvePnP(points, corners, cam_mat, dist_coeff, R, T);
                cout << "Rotation matrix: " << R << endl;
                cout << "Translation matrix: " << T << endl;
                
                //draw 3d axis (Task 5)
                vector<Point2f> out_points;
                Vec3f a = {0,0,3};
                Vec3f b = {3,0,3};
                Vec3f c = {0,3,3};
                vector<Vec3f> temp;
                temp.push_back(a);
                temp.push_back(b);
                temp.push_back(c);

                //Project 3D points
                projectPoints(temp, R, T, cam_mat, dist_coeff, out_points);

                //draw axis
                arrowedLine(frame,  corners[0], out_points[0],Scalar(0,0,255),4,LINE_8,0,0.1);
                arrowedLine(frame, corners[0], corners[2], Scalar(0,255,0),4,LINE_8,0,0.1);
                arrowedLine(frame, corners[0], corners[18], Scalar(255,0,0),4,LINE_8,0,0.1);

                //inserts virtual object (Task 6)
                //also changes the target chessboard (Extension 3)
                draw_shape(frame, R, T, cam_mat, dist_coeff,corners);
                
                // to enable insertion of virtual object on multiple targets in the same scene
                found = 0;
                //remove this chessboard and check for another chessboard
                for(int i=corners[0].y;i<=corners[53].y;i++){
                    for(int j=corners[0].x;j<=corners[53].x;j++){
                        gray.at<uchar>(i,j) = 255;
                    }
                }

                //check for chessboard again
                found = findChessboardCorners(gray, Size(chessboard[0], chessboard[1]), corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                
            }

            imshow("Video", frame);
            
            
            // see if there is a waiting keystroke
            char key = cv::waitKey(10);
            if( key == 'q') {
                break;
            }
            
            //save frame if keypress 's'
            if(key == 's'){
                string str = "/Users/sumeghasinghania/Desktop/CV_Projects/Project4/Resources/Calib_frames/frame_coordinate"+to_string(count)+".jpg";
                imwrite(str, frame);
                count++;
            }

        }

        delete capdev;
    }

    return 0;
}


#endif /* calibration_h */
