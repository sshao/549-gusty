#include <iostream>
#include <iomanip>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <cmath>
#include <cvblob.h>
#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
#include "serial.h"
#include "blobdetect.h"

#define EPSILON 200.0
#define START 1

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

double dist_sqrd(double x1, double y1, double x2, double y2) {
    double x_diff = x1 - x2;
    double y_diff = y1 - y2;

    return (x_diff * x_diff) + (y_diff * y_diff);
}

unsigned int ctr;

int main(int argc, char * argv[])
{
    unsigned int frame_width = 320;
    unsigned int frame_height = 240;
    unsigned int percentage = 5;
    unsigned int min_dist_sqrd_change = 50;
    unsigned int bg_subtractor_history = 50;
    unsigned int bg_subtractor_threshold = 30;
    double area_diff_thresh = 0.30;
    int last_x, last_y;

    const char * filename = "4p-c0.avi";
    
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
        area_diff_thresh = ((double) atoi(argv[7]))/100.0;
    }
    else if (argc == 2) {
        filename = argv[1];
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
        cout << "\tarea difference threshold: if next largest blob is x% "
            << "less than the current largest blob, do not consider any "
            << "more blobs" << endl;
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

    // Set counter
    ctr = 0;
    
    // Set up serial port
    serial_setup();

    // Open camera
    //CvCapture *capture = cvCaptureFromCAM(0);
    CvCapture *capture = cvCaptureFromFile(filename);

    if (!capture) {
	    cerr << "Error opening camera" << endl;
	    return -1;
    }
    
    // Resize window for faster processing
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, frame_width );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, frame_height );

    // See actual window dimensions
    unsigned int real_width = cvGetCaptureProperty(capture, 
        CV_CAP_PROP_FRAME_WIDTH);
    unsigned int real_height = cvGetCaptureProperty(capture, 
        CV_CAP_PROP_FRAME_HEIGHT);

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
    cvMoveWindow("fgmask", real_width, 0);
    
    //cvNamedWindow("bgimg", CV_WINDOW_AUTOSIZE);
    //cvMoveWindow("bgimg", real_width<<1, 0);
    
    //cvNamedWindow("facedetect", CV_WINDOW_AUTOSIZE);
    //cvMoveWindow("facedetect", real_width<<2, 0);
  
    // Grab capture from video
    cvGrabFrame(capture);
    IplImage *img = cvRetrieveFrame(capture);

    CvSize imgSize = cvGetSize(img);
    IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

    BackgroundSubtractorMOG2 bg_model(bg_subtractor_history, 
        bg_subtractor_threshold, bg_subtractor_shadow_detect);

    bool quit = false;
    
    pair<double, double> prev_left;
    pair<double, double> prev_right;

    bool prev_left_valid = false;
    bool prev_right_valid = false;

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
        // FIXME don't actually need this...
        // except for KYL's script
        CvBlobs largest_blobs;
        
        // Now have a list of all the largest blobs. 
        vector<CvBlobs> blob_buckets;
        bool added = false;

        for (int i = 0; i < (blobList.size() * percentage)/100; i++) {
            added = 0;

            largest_blobs.insert(blobList[i]);

            // Print areas
            /*
            cout << "[USED] blob area: " << (*blobList[i].second).area  
                << ", blob bounding rect coords: (" 
                << (*blobList[i].second).minx << ", " << (*blobList[i].second).miny << "), ("
                << (*blobList[i].second).maxx << ", " << (*blobList[i].second).maxy << ")"
                << endl;
            */
        
            // Look for multiple people: if blobs are close together, 
            // group them together

            // If this is the first addition to the list
            if (blob_buckets.size() == 0) {
                CvBlobs new_list;
                blob_buckets.push_back(new_list);
                (blob_buckets.back()).insert(blobList[i]);
            }

            else {
                // Index of bucket that this blob should be added to
                size_t bckt_ind = 0;
                size_t bckt_cur_ind = 0;

                // Current minimum distance to an existing blob
                float bckt_min_dist_y = real_height/2;
                bckt_min_dist_y *= bckt_min_dist_y;
                float bckt_min_dist_x = real_width/8;
                bckt_min_dist_x *= bckt_min_dist_x;
                
                // Iterate through every bucket list
                for (size_t j = 0; j < blob_buckets.size(); j++) {

                    // Iterate down every element of bucket list
                    CvBlobs::iterator it2;
                    for (it2 = blob_buckets[j].begin(); 
                        it2 != blob_buckets[j].end(); it2++) {

                        float bckt_x_sqrd;
                        float bckt_y_sqrd;

                        // calculate x, y distances between current blob 
                        // and all other blobs already bucketed
                        bckt_x_sqrd = ((*(*it2).second).centroid.x - 
                            (*blobList[i].second).centroid.x);
                        bckt_x_sqrd *= bckt_x_sqrd;
                        bckt_y_sqrd = ((*(*it2).second).centroid.y - 
                            (*blobList[i].second).centroid.y);
                        bckt_y_sqrd *= bckt_y_sqrd;

                        // look for closeness in "x" first
                        if (bckt_x_sqrd < bckt_min_dist_x) {
                            // then for closeness in "y"
                            if (bckt_y_sqrd < bckt_min_dist_y) {
                                bckt_ind = j;
                                bckt_min_dist_x = bckt_x_sqrd;
                                bckt_min_dist_y = bckt_y_sqrd;
                                added = true;
                                // if determined to be a candidate for this 
                                // bucket, break out of the bucket and 
                                // move on to the next one
                                break;
                            }
                        }
                    }
                }

                // If never added, make a new bucket
                if (!added) {
                    CvBlobs new_list;
                    blob_buckets.push_back(new_list);
                    (blob_buckets.back()).insert(blobList[i]);
                }
                // Otherwise add to the appropriate bucket
                else {
                    blob_buckets[bckt_ind].insert(blobList[i]);
                }
            }
            
            // If the next blob is less than x% of the current blob's size
            // stop adding blobs and move on to the next step in processing
            if (i != blobList.size()-1) {
                double area_percentage_diff = 
                    (double) (*blobList[i+1].second).area / 
                    (double) (*blobList[i].second).area;
                /*
                printf("%d / %d = %f\n", (*blobList[i+1].second).area, 
                    (*blobList[i].second).area, area_percentage_diff);
                */
                if (area_percentage_diff < area_diff_thresh) {
                    break;
                }
            }
        }

        /*
        cout << "blobs used: " << largest_blobs.size() << endl;

        unsigned int unused_blob_ctr = largest_blobs.size();
        while ((unused_blob_ctr < 50) && (unused_blob_ctr < blobList.size())) {
            cout << "[UNUSED] blob area: " 
                << (*blobList[unused_blob_ctr].second).area  
                << ", blob bounding rect coords: (" 
                << (*blobList[unused_blob_ctr].second).minx << ", " 
                << (*blobList[unused_blob_ctr].second).miny << "), ("
                << (*blobList[unused_blob_ctr].second).maxx << ", " 
                << (*blobList[unused_blob_ctr].second).maxy << ")"
                << endl;
            unused_blob_ctr++;
        }
        */
        
        vector<CvBlobs>::iterator it;
        pair<double, double> left;
        pair<double, double> right;

        bool left_valid = false;
        bool right_valid = false;

        // default values: put "left-most" point at right
        left.first = real_width;
        left.second = 0;

        // default values: put "right-most" point at left
        right.first = 0;
        right.second = 0;

        // Render blobs, and calculate average coordinates for each bucket list
        // on every 10th frame
        if (ctr == 10) {

            // Iterate through every bucket
            for (it = blob_buckets.begin(); it != blob_buckets.end(); it++) {
                // Render all blobs of that bucket
                cvRenderBlobs(labelImg, *it, frame, frame, 
                    CV_BLOB_RENDER_BOUNDING_BOX);
            
                CvBlobs::iterator it2;

                double bckt_avg_x = 0;
                double bckt_avg_y = 0;
                double bckt_weights = 0;

                // Iterate down every bucket and sum up x/y coordinates
                // weighted by area
                for (it2 = (*it).begin(); it2 != (*it).end(); it2++) {
                    bckt_avg_x += (*(*it2).second).area * 
                        (*(*it2).second).centroid.x;
                    bckt_avg_y += (*(*it2).second).area *
                        (*(*it2).second).centroid.y;
                    bckt_weights += (*(*it2).second).area;
                }

                // calculate average x/y coordinates
                if (bckt_weights != 0) {
                    bckt_avg_x /= bckt_weights;
                    bckt_avg_y /= bckt_weights;
                }

                // draw coordinates of averaged point
                cvCircle(frame, cvPoint(bckt_avg_x, bckt_avg_y), 2, 
                    CV_RGB(0, 0, 255), 3, 8, 0);

                // if this is more left than current leftmost point, make left
                if (bckt_avg_x < left.first) {
                    // if the left point has already been assigned at least 
                    // once, check to make sure point being replaced isn't the 
                    // new rightmost point
                    if ((left_valid) && (!right_valid)) {
                        right.first = left.first;
                        right.second = left.second;

                        right_valid = true;
                    }

                    // then replace left point with current point
                    left.first = bckt_avg_x;
                    left.second = bckt_avg_y;

                    left_valid = true;
                }

                // if this is more right than current rightmost point, 
                // make right
                else if (bckt_avg_x > right.first) {
                    // if the right point has already been assigned at least 
                    // once, check to make sure point being replaced isn't 
                    // the new leftmost point
                    if ((right_valid) && (!left_valid)) {
                        left.first = right.first;
                        left.second = right.second;

                        left_valid = true;
                    }

                    // then replace right point with current point
                    right.first = bckt_avg_x;
                    right.second = bckt_avg_y;

                    right_valid = true;
                }
            }

            // reset counter
            ctr = 0;

            // If we have a new left point, save it
            if ((largest_blobs.size() != 0) && (left_valid)) {
                prev_left.first = left.first;
                prev_left.second = left.second;
                prev_left_valid = true;
            }

            // If we have a new right point, save it
            if ((largest_blobs.size() != 0) && (right_valid)) {
                prev_right.first = right.first;
                prev_right.second = right.second;
                prev_right_valid = true;
            }

            // if left assigned, but not right (and right point has been 
            // assigned at least once), check for left > right and decrease 
            // distance between the two
            else if (left_valid && !right_valid && prev_right_valid) {
                // if left point has surpassed previous right point, 
                // switch them
                if (prev_left.first > prev_right.first) {
                    double temp_x = prev_left.first;
                    double temp_y = prev_left.second;

                    prev_left.first = prev_right.first;
                    prev_left.second = prev_right.second;

                    prev_right.first = temp_x;
                    prev_right.second = temp_y;
            
                    // then decrease dist btwn left and right by half by
                    // moving the left point rightwards
                    double x_diff = (prev_left.first - prev_right.first) / 2.0;
                    double y_diff = (prev_left.second - prev_right.second) 
                        / 2.0;

                    prev_left.first -= x_diff;
                    prev_left.second -= y_diff;
                }
                else {
                    // then decrease dist btwn left and right by half by
                    // moving the right point leftwards
                    double x_diff = (prev_right.first - prev_left.first) / 2.0;
                    double y_diff = (prev_right.second - prev_left.second) / 
                        2.0;

                    prev_right.first -= x_diff;
                    prev_right.second -= y_diff;
                }

            }

            // if right assigned, but not left (and left point has been 
            // assigned at least once), check for left > right and decrease 
            // the distance between the two points
            if (right_valid && !left_valid && prev_left_valid) {
                // if right surpassed left, switch them
                if (prev_right.first < prev_left.first) {
                    double temp_x = prev_left.first;
                    double temp_y = prev_left.second;

                    prev_left.first = prev_right.first;
                    prev_left.second = prev_right.second;

                    prev_right.first = temp_x;
                    prev_right.second = temp_y;
            
                    // then decrease dist btwn left and right by half by
                    // moving the rightmost point leftwards
                    double x_diff = (prev_right.first - prev_left.first) / 2.0;
                    double y_diff = (prev_right.second - prev_left.second) / 
                        2.0;

                    prev_right.first -= x_diff;
                    prev_right.second -= y_diff;
                }
                else {
                    // then decrease dist btwn left and right by half by
                    // moving the leftmost point rightwards
                    double x_diff = (prev_left.first - prev_right.first) / 2.0;
                    double y_diff = (prev_left.second - prev_right.second) / 
                        2.0;

                    prev_left.first -= x_diff;
                    prev_left.second -= y_diff;
                }
            }

            // if neither assigned, move both points closer
            if (!right_valid && !left_valid && prev_left_valid &&
                prev_right_valid) {
                double x_diff = (prev_left.first - prev_right.first) / 4.0;
                double y_diff = (prev_left.second - prev_right.second) / 4.0;

                prev_left.first -= x_diff;
                prev_left.second -= y_diff;
            
                prev_right.first += x_diff;
                prev_right.second += y_diff;
            }

            // if the two points are close together, use only left coord
            // invalidate the right coord.
            if (prev_left_valid && prev_right_valid) {
                double x_diff = (prev_left.first - prev_right.first);
                double y_diff = (prev_left.second - prev_right.second);

                double pt_dist = x_diff * x_diff + y_diff * y_diff;

                cout << "pt_dist=" << pt_dist << endl;

                if (pt_dist < EPSILON) {
                    prev_right_valid = false;
                }
            }
        }
        // if no processing to be done this frame
        else {
            ctr++;
        }

        // Draw saved avg point(s) on frame
        if (prev_left_valid) {
            cvCircle(frame, cvPoint(prev_left.first, prev_left.second), 
                5, CV_RGB(255, 0, 0), 3, 8, 0);
        }
        if (prev_right_valid) {
            cvCircle(frame, cvPoint(prev_right.first, prev_right.second), 
                5, CV_RGB(0, 255, 0), 3, 8, 0);
        }

        // Send avg coordinates to arduino
        // TODO check sizes....
        uint16_t left_x_coord, left_y_coord;
        uint16_t right_x_coord, right_y_coord;
        uint32_t left_x_ardu, left_y_ardu;
        uint32_t right_x_ardu, right_y_ardu;
        uint8_t left_ardu_coords, right_ardu_coords;

        // if there are coordinates to send
        if (prev_left_valid || prev_right_valid) {
        // send START bit
        ardu << START;

        // compress left coordinates into a byte
        if (prev_left_valid) {
            left_x_coord = cvRound(prev_left.first);
            left_y_coord = cvRound(prev_left.second);

            left_x_ardu = (left_x_coord * 15) / 320;
            left_y_ardu = (left_y_coord * 15) / 240;

            left_ardu_coords = ((left_x_ardu & 0xf) << 4) | (left_y_ardu & 0xf);
        }
        // compress right coordinates into a byte
        if (prev_right_valid) {
            right_x_coord = cvRound(prev_right.first);
            right_y_coord = cvRound(prev_right.second);

            right_x_ardu = (right_x_coord * 15) / 320;
            right_y_ardu = (right_y_coord * 15) / 240;

            right_ardu_coords = ((right_x_ardu & 0xf) << 4) | (right_y_ardu & 0xf);

            // send
            ardu << right_ardu_coords;
        }
        
        // SEND FIRST COORDINATE
        // if left is valid, send it
        if (prev_left_valid) {
            // send
            ardu << left_ardu_coords;
        }
        // else, send the right coordinate twice
        else {
            ardu << right_ardu_coords;
        }

        // SEND SECOND COORDIANTE
        // if right is valid, send it
        if (prev_right_valid) {
            ardu << right_ardu_coords;
        }
        // else, send the left coordinate twice
        else {
            ardu << left_ardu_coords;
        }

        // cout << "Blob avg coordinates: (" << x_coord << ", " << y_coord << ")" << endl;

        // TODO send STOP?

        // TODO wait for 1 byte from arduino
        while (ardu.get() != 55) {
            // stall
        }

        }

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
        // nts don't need to release largest_blobs bc we inserted straight from
        // the above blobs list

        cout << endl;

        char k = cvWaitKey(10)&0xff;
        switch (k) {
            case 27:
            case 'q':
            case 'Q':
                quit = true;
            break;
            case 'p':
            case 'P':
                while (1) {
                    char p2 = cvWaitKey(10)&0xff;
                    if ((p2 == 'p') || (p2 == 'P')) {
                        break;
                    }
                }
            break;
        }
        
        /*
        if (ctr == 0) {
            sleep(1);
        }
        */
    }

    cvReleaseImage(&frame);
    cvReleaseImage(&img);

    cvDestroyWindow("motion_tracking");
    cvReleaseCapture(&capture);

    /*
    // Release video writer
    if (writer)
        cvReleaseVideoWriter(&writer);
    */

    return 0;
}
