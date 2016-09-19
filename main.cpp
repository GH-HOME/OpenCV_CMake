#include "User_System.h"
#include "CommonOperate.h"

using namespace cv;


int main()
{
	cv::Mat image = imread("test.jpg");
	
	
	vector<cv::Mat>rgb;
	split(image, rgb);

	cv::Mat image_gap;
	addImagegap(image, 100, 100, image_gap);
	imshow("Image_gap", image_gap);
	waitKey();
}