#include <iostream>
#include <iomanip>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <cvblob.h>
#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
#include "serial.h"
#include "facedetect.h"
#include "blobdetect.h"

using namespace cvb;
using namespace cv;
using namespace std;

SerialStream ardu;

/* Compares area of two blobs, returns 1 if p1.area > p2.area */
bool cmpArea(const pair<CvLabel, CvBlob*> &p1, 
    const pair<CvLabel, CvBlob*> &p2)
{
    return p1.second->area > p2.second->area;
}

int main(int argc, char * argv[])
{
    unsigned int frame_width = 320;
    unsigned int frame_height = 240;
    unsigned int percentage = 5;
    unsigned int min_dist_sqrd_change = 5;
    unsigned int bg_subtractor_history = 50;
    unsigned int bg_subtractor_threshold = 16;
    unsigned int area_diff_thresh = 50;
    
    bool bg_subtractor_shadow_detect = false;

    if (argc == 1) {
        cout << "DEFAULT ";
    }
    // No error checking here, probably should...
    else if (argc == 8) {
        frame_width = atoi(argv[1]);
        frame_height = atoi(argv[2]); 
        percentage = atoi(argv[3]);
        min_dist_sqrd_change = atoi(argv[4]);
        bg_subtractor_history = atoi(argv[5]);
        bg_subtractor_threshold = atoi(argv[6]);
        area_diff_thresh = atoi(argv[7]);
    }
    else {
        cout << "Usage: ./blobdetect <width> <height> <percentage> "
            << "<min dist sqrd change> <bg subtractor history> "
            << "<bg subtractor threshold> <area difference threshold>" << endl;
        cout << "\tIf no parameters are specified, " <<
            "will use defaults." << endl;
        cout << "\twidth: resolution width" << endl;
        cout << "\theight: resolution height" << endl;
        cout << "\tpercentage: use x/100 of the largest-area blobs" << endl;
        cout << "\tmin dist sqrd change: minimum change in distance (squared) "
            << "between new point and old point to constitute a change "
            << "in position" << endl;
        cout << "\tbg subtractor history: length of the history in the "
            << "background subtractor" << endl;
        cout << "\tbg subtractor threshold: threshold of the bg/fg "
            << "segmentation" << endl;
        cout << "\tarea difference threshold: if difference between current"
            << " largest blob's area and next larget blob is greater than "
            << " this value, do not consider any more blobs" << endl;
        return 0;
    }
    
    cout << "PARAMETERS: " << endl;
    cout << "\tFrame Width = " << frame_width << endl;
    cout << "\tFrame Height = " << frame_height << endl;
    cout << "\tPercentage of blobs used = " << percentage << endl;
    cout << "\tMin Dist (squared) for change = " 
        << min_dist_sqrd_change << endl;
    cout << "\tBG Subtractor History = " << bg_subtractor_history << endl;
    cout << "\tBG Subtractor Threshold = " << bg_subtractor_threshold << endl;
    cout << "\tArea Difference Threshold = " << area_diff_thresh << endl;
    cout << endl;
    
    // Set up serial port
    serial_setup();

    // Load face detection cascade
    //initializeFaceDetection();

    // Open camera
    CvCapture *capture = cvCaptureFromCAM(0);
    //CvCapture *capture = cvCaptureFromFile("test_video.avi");

    if (!capture) {
	    cerr << "Error opening camera" << endl;
	    return -1;
    }
    
    // Resize window for faster processing
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, frame_width );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, frame_height );

    // Initialize Video writer
    /*
    CvVideoWriter *writer = NULL;
    writer = cvCreateVideoWriter("out.avi", CV_FOURCC('x', 'v', 'i', 'd'),
        25, cvSize(frame_width, frame_height));
    */

    // Create window to render blobs on
    cvNamedWindow("motion_tracking", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("motion_tracking", 0, 0);
    
    cvNamedWindow("fgmask", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("fgmask", frame_width, 0);
    
    //cvNamedWindow("bgimg", CV_WINDOW_AUTOSIZE);
    //cvMoveWindow("bgimg", frame_width<<1, 0);
    
    //cvNamedWindow("facedetect", CV_WINDOW_AUTOSIZE);
    //cvMoveWindow("facedetect", frame_width<<2, 0);
  
    // Grab capture from video
    cvGrabFrame(capture);
    IplImage *img = cvRetrieveFrame(capture);

    CvSize imgSize = cvGetSize(img);
    IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

    // (int history, float varThreshold, bool bShadowDetection=1)
    // length of the history, varThreshold (typical val is 4 sigma, 4*4=16), ...
    BackgroundSubtractorMOG2 bg_model(bg_subtractor_history, 
        bg_subtractor_threshold, bg_subtractor_shadow_detect);

    bool quit = false;
    
    float avg_x = 0;
    float avg_y = 0;
    
    // Main loop
    while (!quit&&cvGrabFrame(capture)) {
        double t = (double) cvGetTickCount();

        img = cvRetrieveFrame(capture);

        cvConvertScale(img, frame, 1, 0);

        Mat fgmask, bgimg;
        Mat img_mat = Mat(img, true);
        // Learn current fg/bg: learning is always on (-1)
        bg_model(img_mat, fgmask, -1);
        // Add to average background image
        //bg_model.getBackgroundImage(bgimg);

        // Show foreground mask
        imshow("fgmask", fgmask);
        // Show the averaged background image
        //imshow("bgimg", bgimg);

        /*
        t = (double) cvGetTickCount () - t;
        printf( "bg/fg segmentation time = %g ms \n",
            t/((double)cvGetTickFrequency() * 1000.) );
        
        t = (double) cvGetTickCount();
        */

        // Convert frame to have 8-bit depth, 1 channel (grayscale)
        IplImage fgmask_ipl = fgmask;
        IplImage* segmentated = cvCreateImage(cvGetSize(&fgmask_ipl), 8, 1);
        cvSetImageCOI(&fgmask_ipl, 1);
        cvCopy(&fgmask_ipl, segmentated);

        // Create image to display for blob tracking
        IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 
            1);

        // Get the blobs 
        CvBlobs blobs;
        unsigned int result = cvLabel(segmentated, labelImg, blobs);

        cout << "blobs found: " << blobs.size() << endl;

        // Sort by largest area
        vector< pair<CvLabel, CvBlob*> > blobList;
        copy(blobs.begin(), blobs.end(), back_inserter(blobList));
        sort(blobList.begin(), blobList.end(), cmpArea);

        // Consider at max only the 1/FRAC_BLOBS_TO_RENDER largest blobs
        // TODO what about the case of multiple people? 
        CvBlobs blobs2;
        double cur_avg_x = 0;
        double cur_avg_y = 0;
        
        for (int i = 0; i < (blobList.size() * percentage)/100; i++) {
            blobs2.insert(blobList[i]);
            cur_avg_x += /*(*blobList[i].second).area * */(*blobList[i].second).centroid.x;
            cur_avg_y += /*(*blobList[i].second).area * */(*blobList[i].second).centroid.y;

            // Print areas
            cout << "[USED] blob area: " << (*blobList[i].second).area  
                << ", blob bounding rect coords: (" 
                << (*blobList[i].second).minx << ", " << (*blobList[i].second).miny << "), ("
                << (*blobList[i].second).maxx << ", " << (*blobList[i].second).maxy << ")"
                << endl;
            
            // If the next blob is way smaller than this one, stop
            if ((i != blobList.size()-1) && 
                (((*blobList[i].second).area - (*blobList[i+1].second).area) > area_diff_thresh)) {
                break;
            }
        }
        if (blobs2.size() != 0) {
            cur_avg_x /= blobs2.size();
            cur_avg_y /= blobs2.size();
        }

        cout << "blobs used: " << blobs2.size() << endl;

        unsigned int unused_blob_ctr = blobs2.size();

        while ((unused_blob_ctr < 50) && (unused_blob_ctr < blobList.size())) {
            cout << "[UNUSED] blob area: " << (*blobList[unused_blob_ctr].second).area  
                << ", blob bounding rect coords: (" 
                << (*blobList[unused_blob_ctr].second).minx << ", " << (*blobList[unused_blob_ctr].second).miny << "), ("
                << (*blobList[unused_blob_ctr].second).maxx << ", " << (*blobList[unused_blob_ctr].second).maxy << ")"
                << endl;
            unused_blob_ctr++;
        }
        
        /*
        t = (double) cvGetTickCount() - t;
        printf( "blob detection/selection time = %g ms\n",
            t/((double)cvGetTickFrequency()*1000.));
        */

        // Render blobs 
        cvRenderBlobs(labelImg, blobs2, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);

        // See if the average point has moved significantly enough to warrant a change.
        double x_diff = cur_avg_x - avg_x;
        double y_diff = cur_avg_y - avg_y;
        double dist_sqrd = x_diff * x_diff + y_diff * y_diff;

        if (dist_sqrd > min_dist_sqrd_change) {
            avg_x = cur_avg_x;
            avg_y = cur_avg_y;
        }

        // Draw avg point on frame
        cvCircle(frame, cvPoint(avg_x, avg_y), 5, CV_RGB(0, 0, 255), 3, 8, 0);

        // face detection
        //detectAndDraw( img_mat );

        // Send avg coordinates to arduino
        uint16_t x_coord, y_coord;
        x_coord = cvRound(avg_x);
        y_coord = cvRound(avg_y);
        cout << "Blob avg coordinates: (" << x_coord << ", " << y_coord << ")" << endl;
        uint32_t x_ardu = (x_coord * 15) / 320;
        uint32_t y_ardu = (y_coord * 15) / 240;

        uint8_t ardu_coords = ((x_ardu & 0xf) << 4) | (y_ardu & 0xf);
        ardu << ardu_coords;

        // Send all coordinates to arduino
        /*
        for (CvBlobs::const_iterator it=blobs2.begin(); it != blobs2.end(); 
            it++) {
            x_coord = cvRound(it->second->centroid.x);
            y_coord = cvRound(it->second->centroid.y);

            //cout << "Blob coordinates: (" << x_coord << ", " << y_coord << ")" << endl;

            uint32_t x_ardu = (x_coord * 15) / 320;
            uint32_t y_ardu = (y_coord * 15) / 240;

            uint8_t ardu_coords = ((x_ardu & 0xf) << 4) | (y_ardu & 0xf);
            ardu << ardu_coords;
        }
        */

        t = (double) cvGetTickCount() - t;
        printf( "total frame processing time = %g ms\n",
            t/((double)cvGetTickFrequency()*1000.));
        
        cvShowImage("motion_tracking", frame);

        // Write to video
        //cvWriteFrame(writer, frame);

        // Release objects
        cvReleaseImage(&labelImg);
        cvReleaseImage(&segmentated);
        cvReleaseBlobs(blobs);

        cout << endl;

        char k = cvWaitKey(10)&0xff;
        switch (k) {
            case 27:
            case 'q':
            case 'Q':
                quit = true;
            break;
        }
    }

    cvReleaseImage(&frame);

    cvDestroyWindow("motion_tracking");
    cvReleaseCapture(&capture);
    // Release video writer
    //if (writer)
    //    cvReleaseVideoWriter(&writer);

    return 0;
}
