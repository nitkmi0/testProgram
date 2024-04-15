// testProgram.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
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
    Mat src = imread("img/lenna.png");
    Mat img1 = imread("img/flower.png");
    Mat dest;
    dest.create(src.size(), src.type());

    //cout << sub(10, 5) << endl;
    //isSameImage(src, dest);
    //testPointer(src, dest);
    testGuiDiff(src, img1);

    return 0;
}