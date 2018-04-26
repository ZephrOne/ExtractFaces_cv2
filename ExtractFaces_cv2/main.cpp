#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <dirent.h>
#include <list>
#include <thread>

#define FRAME_PRE_SENCOND 1 //隔多少帧获取一次frame
#define ACTIVE_CAM true //决定使用视频还是摄像头
#define DRAW true //显示人脸检测框
#define VIDEO_PATH "/Users/Bevis/Desktop/Dev/C++ Project/ExtractFaces_cv2/video/video.mp4" //视频文件路径
#define IMAGE_PATH "/Users/Bevis/Desktop/1/" //图片文件夹路径

using namespace cv;
using namespace std;

//检测人脸
int detect( UMat & img, int &count, list<Mat> &FACES, int &i) {
    Mat show;
    if(i==FRAME_PRE_SENCOND){
        i=0;
    CascadeClassifier hc;
    string path;
    //配置文件，一定要有
    //path = "/Users/Bevis/Desktop/Dev/C++ Project/ExtractFaces_cv2/ExtractFaces_cv2/Face_cascade.xml";
    path = "/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml";
    if ( !hc.load( path ) ) {
        printf("Failed to load Haar cascade\n");
        return -1;
    }
    cvtColor(img,img, CV_BGR2GRAY);
    vector<Rect> faces;
    faces.clear();
    
    hc.detectMultiScale(img, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    
        
        show =img.getMat(ACCESS_RW);
    for ( size_t i=0; i<faces.size(); i++ ) {
        
        //Safely expand the ROI.
        Rect ROI;
        ROI.x = (faces[i].x-20>0) ? faces[i].x-20 : 0;
        ROI.width = (faces[i].width+40<img.cols) ? faces[i].width+40 : img.cols;
        ROI.y = (faces[i].y-25>0) ? faces[i].y-25 : 0;
        ROI.height = (faces[i].height+50<img.rows) ? faces[i].width+50 : img.rows;
        
        if (DRAW) {
            rectangle(show, ROI, Scalar(0,255,0));
            putText(img, "Detect Face!",
                    Point(20, 50),
                    FONT_HERSHEY_COMPLEX, 1, // font face and scale
                    Scalar(255, 255, 255), // white
                    1, LINE_AA); // line thickness and type
        }
        
        
        Mat imgROI;
        try {
            imgROI = img(ROI).getMat(ACCESS_RW);
        } catch (cv::Exception & ex) {
            cout<<"ROI:"<<ex.what();
            return -1;
        }
        
        cv::resize(imgROI, imgROI, Size(244,244));//修改图片分辨率
        FACES.push_back(imgROI);
        imshow("Frame", imgROI);
        cout<< FACES.size() <<" face(s) collected"<<endl;
        }
       
        
        //debug
        //printf("x: %d - %d \n",faces[i].x,faces[i].x+faces[i].width);
        //printf("y: %d - %d \n",faces[i].y,faces[i].y+faces[i].height);
        /*
        //文件名
        stringstream name("Capture");
        name << count;
        name << i+1;
        name << ".jpg";
        string destfilename = name.str();
        
        //生成图片文件
        try {
            imshow("test", imgROI);
            imwrite(destfilename, Mat(imgROI));
        } catch (cv::Exception &ex) {
            cerr<<ex.what();
        }
        name.ignore();
         */
        
    }
   else show = img.getMat(ACCESS_RW);
    imshow("CAM", show);
    
    //printf("%lu face(s) extracted\n", faces.size());
    
    return 0;
}

//图片文件搜索
bool traverseFile(list<Mat> &FACES){
    struct dirent *dirp;
    DIR* dir = opendir(IMAGE_PATH);
    int i=FRAME_PRE_SENCOND;
    int count = 1;
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type == DT_REG) {
            // 文件
            string directionToimage = IMAGE_PATH;
            directionToimage += dirp->d_name;
            cout<<dirp->d_name<<endl;
            Mat img;
            img = imread(directionToimage);
            UMat x = img.getUMat(ACCESS_RW);
            if ( !img.data ) {
                cout<<"File not found\n";
                continue;
            }
            resize(img, img, Size(672,504));
            detect(x,count,FACES, i);
            waitKey();
        } else if (dirp->d_type == DT_DIR) {
            // 文件夹
        }
    }
    
    closedir(dir);
    return true;
}

//使用视频或者摄像头
bool VideoCapture_cv(list<Mat> &FACES){
    UMat frame;
    VideoCapture video;
    int i=0, count = 1;
    
    if (ACTIVE_CAM) {
        video.open(0);
    }else
        video.open(VIDEO_PATH); //cout<<video.get(CV_CAP_PROP_FRAME_COUNT);
    
    video.set(CAP_PROP_FPS, 30);
    
    while (true) {
        video >> frame;
        if (frame.empty()) {
            return 0;
        }
        
        //resize(frame, frame, Size(640,360));
        resize(frame, frame, Size(),0.5,0.5);
        
            //video >> frame;
        count++;
            //imshow("video", frame);
        detect(frame, count, FACES,i);
        if(waitKey(10) >= 0)
            break;
        i++;
        //std::this_thread::sleep_for(chrono::nanoseconds(1000*1000*500));
    }
    
    return true;
}

bool ImageCapture_cv(list<Mat> &FACES){//提取一个文件夹下所有图片文件中的人脸
    traverseFile(FACES);
    return true;
}

int main(int argc, char* argv[]) {
    
    list<Mat> FACES;
    //ImageCapture_cv(FACES);
    VideoCapture_cv(FACES);
    for(auto & item : FACES)
    {
        imshow("Hahahahahaha",item);
        waitKey();
    }
    
    return 0;
}




















/*
int main()
{
    cout << "Built with OpenCV " << CV_VERSION << endl;
    Mat image;
    VideoCapture capture;
    capture.open(0);
    if(capture.isOpened())
    {
        cout << "Capture is opened" << endl;
        for(;;)
        {
            capture >> image;
            if(image.empty())
                break;
            drawText(image);
            imshow("Sample", image);
            if(waitKey(10) >= 0)
                break;
        }
    }
    else
    {
        cout << "No capture" << endl;
        image = Mat::zeros(480, 640, CV_8UC1);
        drawText(image);
        imshow("Sample", image);
        waitKey(0);
    }
    return 0;
}

void drawText(Mat & image)
{
    putText(image, "Hello OpenCV",
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}

//调用camera函数
 
 CvCapture *cap;
 IplImage *frame;
 const char ESC = 27;
 if ((cap = cvCreateCameraCapture(0)) != 0)
 {
 cvNamedWindow("Camera");
 while ((frame = cvQueryFrame(cap)) != 0 &&
 cvWaitKey(20) != ESC)
 {
 frame = cvQueryFrame(cap);
 cvShowImage("Camera", frame);
 }
 cvDestroyWindow("Camera");
 }
 */

/*image Capture
 cvNamedWindow("视频显示",CV_WINDOW_AUTOSIZE);
 
 CvCapture* capture = cvCreateFileCapture("/Users/Bevis/Desktop/Dev/C++ Project/ExtractFaces_cv2/video.mp4");
 
 IplImage* frame;
 while(1)
 {
 frame = cvQueryFrame(capture);
 if (!frame) break;
 cvShowImage("视频显示",frame);
 char c = cvWaitKey(33);
 if(c==27)break;
 }
 cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,1.0);
 IplImage* image = cvQueryFrame(capture);
 
 cvShowImage("T", image);
 waitKey();
 
 cvReleaseCapture(&capture);
 cvDestroyWindow("视频显示");
 */




