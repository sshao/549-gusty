#ifndef _BLOBDETECT_H_
#define _BLOBDETECT_H_

using namespace cvb;
using namespace std;

bool cmpArea(const pair<CvLabel, CvBlob*> &p1, 
    const pair<CvLabel, CvBlob*> &p2);

#endif
