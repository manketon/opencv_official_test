#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <vector>
using namespace cv;
using namespace std;
//https://blog.csdn.net/liangfei868/article/details/124680559
//https://github.com/SFNCY/Steger/tree/master/RailTurnout/RailTurnout
int test_steger_2(std::string& str_err_reason)
{//得到的中心线有很多离散的点
	std::string str_src_img_path = "guangxian.jpg";
	std::cout << "请输入源二值图路径:";
	std::cin >> str_src_img_path;
	cv::Mat img0 = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
	imshow("原图", img0);
	Mat img = img0.clone();

	//高斯滤波
	img.convertTo(img, CV_32FC1);
	GaussianBlur(img, img, Size(0, 0), 6, 6);

	//一阶偏导数
	Mat m1, m2;
	m1 = (Mat_<float>(1, 2) << 1, -1);  //x偏导
	m2 = (Mat_<float>(2, 1) << 1, -1);  //y偏导

	Mat dx, dy;
	filter2D(img, dx, CV_32FC1, m1);
	filter2D(img, dy, CV_32FC1, m2);

	//二阶偏导数
	Mat m3, m4, m5;
	m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
	m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
	m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导

	Mat dxx, dyy, dxy;
	filter2D(img, dxx, CV_32FC1, m3);
	filter2D(img, dyy, CV_32FC1, m4);
	filter2D(img, dxy, CV_32FC1, m5);

	//hessian矩阵
	double maxD = -1;
	int imgcol = img.cols;
	int imgrow = img.rows;
	vector<double> Pt;
	for (int i = 0; i < imgcol; i++)
	{
		for (int j = 0; j < imgrow; j++)
		{
			if (img0.at<uchar>(j, i) > 200)
			{
				Mat hessian(2, 2, CV_32FC1);
				hessian.at<float>(0, 0) = dxx.at<float>(j, i);
				hessian.at<float>(0, 1) = dxy.at<float>(j, i);
				hessian.at<float>(1, 0) = dxy.at<float>(j, i);
				hessian.at<float>(1, 1) = dyy.at<float>(j, i);

				Mat eValue;
				Mat eVectors;
				eigen(hessian, eValue, eVectors);

				double nx, ny;
				double fmaxD = 0;
				if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //求特征值最大时对应的特征向量
				{
					nx = eVectors.at<float>(0, 0);
					ny = eVectors.at<float>(0, 1);
					fmaxD = eValue.at<float>(0, 0);
				}
				else
				{
					nx = eVectors.at<float>(1, 0);
					ny = eVectors.at<float>(1, 1);
					fmaxD = eValue.at<float>(1, 0);
				}

				double t = -(nx * dx.at<float>(j, i) + ny * dy.at<float>(j, i)) / (nx * nx * dxx.at<float>(j, i) + 2 * nx * ny * dxy.at<float>(j, i) + ny * ny * dyy.at<float>(j, i));

				if (fabs(t * nx) <= 0.5 && fabs(t * ny) <= 0.5)
				{
					Pt.push_back(i);
					Pt.push_back(j);
				}
			}
		}
	}

	for (int k = 0; k < Pt.size() / 2; k++)
	{
		Point rpt;
		rpt.x = Pt[2 * k + 0];
		rpt.y = Pt[2 * k + 1];
		circle(img0, rpt, 1, Scalar(0, 0, 255));
	}
	//namedWindow("result", WINDOW_NORMAL);
	imshow("result", img0);
	waitKey(0);
	return 0;
}
int test_steger_1(std::string& str_err_reason);

int test_steger(std::string& str_err_reason)
{
	int ret = -1;
	while (true)
	{
		std::cout << "0:退出" << std::endl;
		std::cout << "1、方法1" << std::endl;
		std::cout << "2、方法2" << std::endl;
		std::cout << "3、方法3" << std::endl;
		std::cout << "请输入你的选择:";
		int nChoise = 0;
		std::cin >> nChoise;
		switch (nChoise)
		{
		case 0:
			std::cout << "成功退出" << std::endl;
			return 0;
		case 1:
			ret = test_steger_1(str_err_reason);
			break;
		case 2:
			ret = test_steger_2(str_err_reason);
			break;
		case 3:
			break;
		default:
			break;
		}
	}
	return 0;
}
