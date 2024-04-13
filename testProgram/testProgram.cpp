// testProgram.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <fstream>
#include <iostream>
#include <opencp.hpp>
#pragma comment(lib,"opencp.lib")
#include<immintrin.h>


using namespace std;
using namespace cv;
using namespace cp;

void isComplateSame(Mat src, Mat ref)
{
    bool is = true;
    for (int c = 0; c < 3; c++)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                if (src.at<Vec3b>(j, i)[c] != ref.at<Vec3b>(j, i)[c]) is = false;
            }
        }
    }
    if (is) { printf("Complete Same\n"); }
    else { printf("Not Same\n"); }
}

//#define POINTER
//#define DIFF
#define DIFF_ORIGINAL

int main()
{
#ifdef POINTER

    Mat src, dest, src_yuv;
    vector<Mat> splitimg;
    src = imread("img/lenna.png");
    dest.create(src.size(), src.type());

    cv::cvtColor(src, src_yuv, COLOR_BGR2YUV);
    split(src_yuv, splitimg);
    splitimg[0].convertTo(splitimg[0], CV_32FC1);
    imshow("splitimg[0]_before", splitimg[0] / 255);

    // 処理(splitimg[0])
    for (int j = 0; j < splitimg[0].rows; j++)
    {
        float* ptr = splitimg[0].ptr<float>(j, 100);

        //for (int i = 0; i < splitimg[0].cols; i++)
        {
            ptr[100] = 0;
            //*ptr++ = 128;

            /*ptr[1] = 0;
            ptr[2] = 0;*/
            //*ptr = 0;
            ptr += splitimg[0].cols;
        }
    }
    imshow("splitimg[0]_after", splitimg[0] / 255);


    splitimg[0].convertTo(splitimg[0], CV_8UC1);
    splitimg[0] = Mat(splitimg[0], Rect(0, 0, splitimg[1].cols, splitimg[1].rows));
    cv::merge(splitimg, dest);
    cv::cvtColor(dest, dest, COLOR_YUV2BGR);

    imshow("source", src);
    imshow("src_convertToYUV", src_yuv);
    imshow("destination", dest);
    waitKey(0);

#endif
#ifdef DIFF
    Mat img1 = imread("img/lenna.png");
    Mat img2 = imread("img/flower.png");
    resize(img2, img2, Size(img1.cols, img1.rows));
    imshow("img2_resized", img2);
    //waitKey(0);

    cp::guiDiff(img1, img2);
#endif

#ifdef DIFF_ORIGINAL
    Mat img1 = imread("img/lenna.png");
    Mat img2 = imread("img/flower.png");

    isComplateSame(img1, img2);
    isComplateSame(img1, img1);
    
#endif
}