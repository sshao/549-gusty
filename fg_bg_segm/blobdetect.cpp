#include <iostream>
#include <iomanip>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <cvblob.h>
#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
#include "serial.h"
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
    unsigned int min_dist_sqrd_change = 3000;

    if (argc == 1) {
        cout << "DEFAULT ";
    }
    // No error checking here, probably should...
    else if (argc == 5) {
        frame_width = atoi(argv[1]);
        frame_height = atoi(argv[2]); 
        percentage = atoi(argv[3]);
        min_dist_sqrd_change = atoi(argv[4]);
    }
    else {
        cout << "Usage: ./blobdetect <width> <height> <percentage> "
            << "<min dist sqrd change>" << endl;
        cout << "\tIf no parameters are specified, " <<
            "will use defaults." << endl;
        cout << "\twidth: resolution width" << endl;
        cout << "\theight: resolution height" << endl;
        cout << "\tpercentage: use x/100 of the largest-area blobs" << endl;
        cout << "\tmin dist sqrd change: minimum change in distance (squared) "
            << "between new point and old point to constitute a change "
            << "in position" << endl;
        return 0;
    }
    
    cout << "PARAMETERS: " << endl;
    cout << "\tFrame Width = " << frame_width << endl;
    cout << "\tFrame Height = " << frame_height << endl;
    cout << "\tPercentage of blobs used = " << percentage << endl;
    cout << "\tMin Dist (squared) for change = " 
        << min_dist_sqrd_change << endl;
    
    // Set up serial port
    serial_setup();

    // Open camera
    CvCapture *capture = cvCaptureFromCAM(0);

    if (!capture) {
	    cerr << "Error opening camera" << endl;
	    return -1;
    }
    
    // Resize window for faster processing
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, frame_width );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, frame_height );

    // Create window to render blobs on
    namedWindow("motion_tracking", CV_WINDOW_AUTOSIZE);
  
    // Grab capture from video
    cvGrabFrame(capture);
    IplImage *img = cvRetrieveFrame(capture);

    CvSize imgSize = cvGetSize(img);
    IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

    BackgroundSubtractorMOG2 bg_model;

    bool quit = false;
    
    float avg_x = 0;
    float avg_y = 0;
    
    // Main loop
    while (!quit&&cvGrabFrame(capture)) {
        img = cvRetrieveFrame(capture);

        cvConvertScale(img, frame, 1, 0);

        Mat fgmask, bgimg;
        // Learn current fg/bg: learning is always on (-1)
        bg_model(Mat(img, true), fgmask, -1);
        // Add to average background image
        bg_model.getBackgroundImage(bgimg);

        // Show foreground mask
        imshow("fgmask", fgmask);
        // Show the averaged background image
        imshow("bgimg", bgimg);

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

        // Sort by largest area
        vector< pair<CvLabel, CvBlob*> > blobList;
        copy(blobs.begin(), blobs.end(), back_inserter(blobList));
        sort(blobList.begin(), blobList.end(), cmpArea);

        // Grab only the 1/FRAC_BLOBS_TO_RENDER largest blobs
        // TODO what about the case of multiple people? 
        CvBlobs blobs2;
        unsigned int cur_avg_x = 0;
        unsigned int cur_avg_y = 0;
        // FIXME careful with the division/mult here...
        cout << "using " << ((blobList.size() * percentage)/100) 
            << " blobs" << endl;
        for (int i = 0; i < (blobList.size() * percentage)/100; i++) {
            blobs2.insert(blobList[i]);
            cur_avg_x += /*(*blobList[i].second).area * */(*blobList[i].second).centroid.x;
            cur_avg_y += /*(*blobList[i].second).area * */(*blobList[i].second).centroid.y;
            // Print out areas
            //cout << "[" << blobList[i].first << "] -> " << (*blobList[i].second).area << endl;
        }
        if (blobs2.size() != 0) {
            cur_avg_x /= blobs2.size();
            cur_avg_y /= blobs2.size();
        }

        // Render blobs 
        cvRenderBlobs(labelImg, blobs2, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);

        // See if the average point has moved significantly enough to warrant a change.
        unsigned int x_diff = cur_avg_x - avg_x;
        unsigned int y_diff = cur_avg_y - avg_y;
        unsigned int dist_sqrd = x_diff * x_diff + y_diff * y_diff;
        if (dist_sqrd > min_dist_sqrd_change) {
            avg_x = cur_avg_x;
            avg_y = cur_avg_y;
        }

        // Draw avg point on frame
        cvCircle(frame, cvPoint(avg_x, avg_y), 5, CV_RGB(0, 0, 255), 3, 8, 0);
//        cvSetAt(frame, cvScalar(255, 0, 0, 0), avg_y, avg_x);

        uint16_t x_coord, y_coord;

        // Send avg coordinates to arduino
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

        cvShowImage("motion_tracking", frame);

        // Release objects
        cvReleaseImage(&labelImg);
        cvReleaseImage(&segmentated);
        cvReleaseBlobs(blobs);

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

    return 0;
}
