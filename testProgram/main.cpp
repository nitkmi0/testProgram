﻿// testProgram.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//
#include <iostream>
#include <opencp.hpp>
#include "testFunction.hpp"

#pragma comment(lib,"opencp.lib")

using namespace std;
using namespace cv;
using namespace cp;


int main()
{
    //Mat src = imread("img/lenna.png");
    //imshow("src", src);
    //Mat img1 = imread("img/flower.png");
    /*Mat dest;
    dest.create(src.size(), src.type());*/

    //cout << sub(10, 5) << endl;
    //isSameImage(src, dest);
    //testPointer(src, dest);
    //testGuiDiff(src, img1);

    // test adaptive
    /*Mat src = imread("datasets/sawtooth/im0.png");
    Mat dispMap = imread("datasets/sawtooth/disp2.png");
    Mat dest(src.size(), src.type());

    Mat dispMap32F(dispMap.size(), CV_32FC1);
    dispMap.convertTo(dispMap32F, CV_32FC1, 1.f / 255.f);
    
    testAdaptiveGaussianFilter(src, dispMap32F, dest);*/

    // test Plot
    testPlot();

    return 0;
}