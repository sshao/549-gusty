#include <iostream>
#include <stdio.h>

#include "cv.h"
#include "highgui.h"

#include "facedetect.h"

using namespace std;
using namespace cv;

// Cascade info
String cascadeName = "haarcascade_frontalface_alt.xml";
CascadeClassifier cascade;

void initializeFaceDetection( void ) {
    if (!cascade.load(cascadeName)) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
    }
}

void detectAndDraw( Mat& img )
{
    int i = 0;
    double t = 0;
    vector<Rect> faces;
    Mat gray, smallImg( cvRound (img.rows), cvRound(img.cols), 
        CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();

    cascade.detectMultiScale( smallImg, faces, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, Size(30, 30) );
    /*
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        // Default: 30, 30
        Size(50, 50) );
    */
    t = (double)cvGetTickCount() - t;
    printf( "face detection time = %g ms\n", 
        t/((double)cvGetTickFrequency()*1000.) );

    for( ; i < faces.size(); i++) 
    {
        vector<Rect> nestedObjects;
        Point center;
        int radius;
        center.x = cvRound((faces[i].x + faces[i].width*0.5));
        center.y = cvRound((faces[i].y + faces[i].height*0.5));
        radius = cvRound((faces[i].width + faces[i].height)*0.25);
        circle( img, center, radius, CV_RGB(0,0,255), 3, 8, 0 );

        //cout << "(" << center.x << "," << center.y << ")" << endl;
/*
        // Send x coordinates, MSB and LSB respectively
        ardu << (center.x >> 8) && 0xff;
        ardu << center.x && 0xff;
        // Sned y coordinates
        ardu << (center.y >> 8) && 0xff;
        ardu << center.y && 0xff;
*/
        // Label coordinates
        /*
        // TODO better way of allocating space
        char coords[20];
        Point text_pos;
        text_pos.x = center.x + 5;
        text_pos.y = center.y + 22;
        sprintf(coords, "(%d, %d)", center.x, center.y);
        putText( img, coords, text_pos, FONT_HERSHEY_SIMPLEX, 0.55,
            color, 2, 8, 0);
        */
    }
    cv::imshow("facedetect", img );
}
