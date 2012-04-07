#ifndef _FACEDETECT_H_
#define _FACEDETECT_H_

using namespace std;
using namespace cv;

extern String cascadeName;
extern CascadeClassifier cascade;

void initializeFaceDetection( void );
void detectAndDraw( Mat& img  );

#endif
