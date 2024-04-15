#include "testFunction.hpp"
#include <iostream>
#include <opencp.hpp>

#pragma comment(lib,"opencp.lib")

int sub(int x, int y)
{
    return x - y;
}

void testGuiDiff(Mat& src, Mat& ref)
{
    resize(ref, ref, Size(src.cols, src.rows));
    imshow("ref_resized", ref);

    cp::guiDiff(src, ref);
}

void testPointer(Mat& src, Mat& dest)
{
    Mat src_yuv;
    vector<Mat> splitimg;
    dest.create(src.size(), src.type());

    cv::cvtColor(src, src_yuv, COLOR_BGR2YUV);
    split(src_yuv, splitimg);
    splitimg[0].convertTo(splitimg[0], CV_32FC1);
    imshow("splitimg[0]_before", splitimg[0] / 255);

    // èàóù(splitimg[0])
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
}

void isSameImage(Mat& src, Mat& ref)
{
    bool isSame = true;
    for (int c = 0; c < 3; c++)
    {
        for (int j = 0; j < src.rows; j++)
        {
            for (int i = 0; i < src.cols; i++)
            {
                if (src.at<Vec3b>(j, i)[c] != ref.at<Vec3b>(j, i)[c]) isSame = false;
            }
        }
    }
    if (isSame) cout << "Complete Same" << endl;
    else cout << "Not Same" << endl;
}

void AdaptiveGaussianFilter(cv::Mat& src, cv::Mat& dispMap, cv::Mat& dest, const int r, const float sigma_base, const float inc_sigma, const float inforcus_disp)
{
    Mat srcBorder;
    copyMakeBorder(src, srcBorder, r, r, r, r, BORDER_REFLECT);

    const int kernel_size = (2 * r + 1) * (2 * r + 1);
    int* space_offset = (int*)malloc(sizeof(int) * kernel_size);
    float* space_pos = (float*)malloc(sizeof(float) * kernel_size);

    int count = 0;
    for (int y = -r; y <= r; y++)
    {
        for (int x = -r; x <= r; x++)
        {
            space_pos[count] = (float)(x * x + y * y);
            space_offset[count++] = y * srcBorder.cols + x;
        }
    }

    const int cn = 3;
#pragma omp parallel for schedule (dynamic)
    for (int y = 0; y < dest.rows; y++)
    {
        const float* sp = srcBorder.ptr<float>(y + r) + r * cn;
        const uchar* mp = dispMap.ptr<uchar>(y);
        float* dp = dest.ptr<float>(y);
        for (int x = 0; x < dest.cols * cn; x += cn)
        {
            const float* spx = sp + x;
            const uchar* mpx = mp + x / cn;
            float sum_b = 0.f;
            float sum_g = 0.f;
            float sum_r = 0.f;
            float sum_w = 0.f;
            const float sigma = sigma_base + abs((float)*mpx - inforcus_disp) * inc_sigma;
            for (int k = 0; k < kernel_size; k++)
            {
                const float* spk = spx + space_offset[k] * 3;
                float src_ref_b = spk[0];
                float src_ref_g = spk[1];
                float src_ref_r = spk[2];

                float w = exp(space_pos[k] / (-2 * sigma * sigma));

                sum_b += src_ref_b * w;
                sum_g += src_ref_g * w;
                sum_r += src_ref_r * w;
                sum_w += w;
            }
            float* dpx = dp + x;
            dpx[0] = sum_b / sum_w;
            dpx[1] = sum_g / sum_w;
            dpx[2] = sum_r / sum_w;
        }
    }
    free(space_pos);
    free(space_offset);
}