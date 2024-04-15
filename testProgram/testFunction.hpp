#pragma once
#include <opencp.hpp>
#pragma comment(lib, "opencp.lib")

using namespace cv;
using namespace cp;
using namespace std;

int sub(int x, int y);
void testGuiDiff(Mat& src, Mat& dest);
void isSameImage(Mat& src, Mat& ref);
void testPointer(Mat& src, Mat& dest);
void AdaptiveGaussianFilter(cv::Mat& src, cv::Mat& dispMap, cv::Mat& dest, const int r, const float sigma_base, const float inc_sigma, const float inforcus_disp);