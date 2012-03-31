#include <iostream>
#include <iomanip>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <cvblob.h>
#include <SerialStream.h>

#define PORT "/dev/ttyACM0"
#define BAUDRATE (SerialStreamBuf::BAUD_9600)

using namespace cvb;
using namespace cv;
using namespace LibSerial;
using namespace std;

SerialStream ardu;

// Setup serial communication with Arduino
void serial_setup ( void ) {
    ardu.Open(PORT);
    ardu.SetBaudRate(BAUDRATE);
    ardu.SetCharSize(SerialStreamBuf::CHAR_SIZE_8);
}

int main(int argc, char * argv[])
{
    unsigned int frame_width = 320;
    unsigned int frame_height = 240;
    // HSV threshold values for red. 
    CvScalar hsv_min = cvScalar(170, 50, 170, 0);
    CvScalar hsv_max = cvScalar(256, 180, 256, 0);

    if (argc == 1) {
        cout << "DEFAULT ";
    }
    // No error checking here, probably should...
    else if (argc == 9) {
        frame_width = atoi(argv[1]);
        frame_height = atoi(argv[2]); 
        for (int i = 0; i < 3; i++) { 
            hsv_min.val[i] = atoi(argv[i+3]);
            hsv_max.val[i] = atoi(argv[i+6]);
        }
    }
    else {
        cout << "Usage: ./blobdetect <width> <height> " <<
            "<min threshold H> <min threshold S> <min threshold V> " <<
            "<max threshold H> <max threhold S> <max threshold V> " << endl;
        cout << "\tIf no parameters are specified, " <<
            "will use defaults." << endl;
        cout << "\twidth: resolution width" << endl;
        cout << "\theight: resolution height" << endl;
        cout << "\tmin threshold H,S,V: minimum threshold values of color " <<
            "being detected in HSV" << endl;
        cout << "\tmax threwhold H,S,V: maximum threshold values of color " <<
            "being detected in HSV" << endl;
        cout << "\t" << endl;
        return 0;
    }
    
    cout << "PARAMETERS: " << endl;
    cout << "\tFrame Width = " << frame_width << endl;
    cout << "\tFrame Height = " << frame_height << endl;
    cout << "\tMin threshold (HSV) = (" << hsv_min.val[0] << ", " << hsv_min.val[1] << ", " << hsv_min.val[2] << ")" << endl;
    cout << "\tMax threshold (HSV) = (" << hsv_max.val[0] << ", " << hsv_max.val[1] << ", " << hsv_max.val[2] << ")" << endl;

    serial_setup();

    CvCapture *capture = cvCaptureFromCAM(0);

    if (!capture) {
	    cerr << "Error opening camera" << endl;
	    return -1;
    }
    
    // Resize window for faster processing
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, frame_width );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, frame_height );

    namedWindow("red_object_tracking", CV_WINDOW_AUTOSIZE);
  
    cvGrabFrame(capture);
    IplImage *img = cvRetrieveFrame(capture);

    CvSize imgSize = cvGetSize(img);

    IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);
    IplImage *hsvframe = cvCreateImage(imgSize, img->depth, img->nChannels);

    IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, 
        CV_SHAPE_RECT, NULL);

    unsigned int blobNumber = 0;

    bool quit = false;
    
    while (!quit&&cvGrabFrame(capture)) {
        img = cvRetrieveFrame(capture);

        cvConvertScale(img, frame, 1, 0);

        IplImage *segmentated = cvCreateImage(imgSize, 8, 1);
        IplImage *segmentated1 = cvCreateImage(imgSize, 8, 1);
    
        // Detecting red pixels:
        // (This is very slow, use direct access better...)
        /*
        for (unsigned int j=0; j<imgSize.height; j++)
            for (unsigned int i=0; i<imgSize.width; i++) {
	            CvScalar c = cvGet2D(frame, j, i);

	            double b = ((double)c.val[0])/255.;
	            double g = ((double)c.val[1])/255.;
	            double r = ((double)c.val[2])/255.;
	            unsigned char f = 255*((r>0.2+g)&&(r>0.2+b));

	            cvSet2D(segmentated, j, i, CV_RGB(f, f, f));
            }

        cvMorphologyEx(segmentated, segmentated, NULL, morphKernel, 
            CV_MOP_OPEN, 1);
        */

        //cvShowImage("segmentated", segmentated);
        
        /* NEW ATTEMPT AT COLOR SEPARATION */
        // HSV: (hue, saturation, value)
        // In HSV, red hue wraps around: we may need 2 ranges. If we do, then
        // call cvInRangeS twice (once for each range) and then cvOr 
        // the results.
        // CvScalar hsv_min1 = cvScalar(0, 50, 170, 0);
        // CvScalar hsv_max1 = cvScalar(10, 180, 256, 0);

        cvCvtColor(frame, hsvframe, CV_BGR2HSV);
        //cvInRangeS(hsvframe, hsv_min1, hsv_max1, segmentated);
        cvInRangeS(hsvframe, hsv_min, hsv_max, segmentated); // change to segmentated1 if using 2 ranges
        //cvOr(segmentated, segmentated, segmentated1);

        IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 
            1);

        CvBlobs blobs;
        // Get the blobs
        unsigned int result = cvLabel(segmentated, labelImg, blobs);
        // PICK A BLOB: the largeset one
        cvFilterByLabel(blobs, cvGreaterBlob(blobs));
        cvRenderBlobs(labelImg, blobs, frame, frame);

        uint16_t x_coord, y_coord;

        for (CvBlobs::const_iterator it=blobs.begin(); it != blobs.end(); 
            it++) {
            x_coord = cvRound(it->second->centroid.x);
            y_coord = cvRound(it->second->centroid.y);

            cout << "Blob coordinates: (" << x_coord << ", " << y_coord << ")" << endl;

            // Send coordinates to arduino
            uint32_t x_ardu = (x_coord * 15) / 320;
            uint32_t y_ardu = (y_coord * 15) / 240;

            uint8_t ardu_coords = ((x_ardu & 0xf) << 4) | (y_ardu & 0xf);
            ardu << ardu_coords;
        }

        cvShowImage("red_object_tracking", frame);

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

    cvReleaseStructuringElement(&morphKernel);
    cvReleaseImage(&frame);

    cvDestroyWindow("red_object_tracking");
    cvReleaseCapture(&capture);

    return 0;
}
