#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <vector>
using namespace cv;
using namespace std;
//https://blog.csdn.net/liangfei868/article/details/124680559
//https://github.com/SFNCY/Steger/tree/master/RailTurnout/RailTurnout
int test_steger_2(std::string& str_err_reason)
{//�õ����������кܶ���ɢ�ĵ�
	std::string str_src_img_path = "guangxian.jpg";
	std::cout << "������Դ��ֵͼ·��:";
	std::cin >> str_src_img_path;
	cv::Mat img0 = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
	imshow("ԭͼ", img0);
	Mat img = img0.clone();

	//��˹�˲�
	img.convertTo(img, CV_32FC1);
	GaussianBlur(img, img, Size(0, 0), 6, 6);

	//һ��ƫ����
	Mat m1, m2;
	m1 = (Mat_<float>(1, 2) << 1, -1);  //xƫ��
	m2 = (Mat_<float>(2, 1) << 1, -1);  //yƫ��

	Mat dx, dy;
	filter2D(img, dx, CV_32FC1, m1);
	filter2D(img, dy, CV_32FC1, m2);

	//����ƫ����
	Mat m3, m4, m5;
	m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //����xƫ��
	m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //����yƫ��
	m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //����xyƫ��

	Mat dxx, dyy, dxy;
	filter2D(img, dxx, CV_32FC1, m3);
	filter2D(img, dyy, CV_32FC1, m4);
	filter2D(img, dxy, CV_32FC1, m5);

	//hessian����
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
				if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //������ֵ���ʱ��Ӧ����������
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
		std::cout << "0:�˳�" << std::endl;
		std::cout << "1������1" << std::endl;
		std::cout << "2������2" << std::endl;
		std::cout << "3������3" << std::endl;
		std::cout << "���������ѡ��:";
		int nChoise = 0;
		std::cin >> nChoise;
		switch (nChoise)
		{
		case 0:
			std::cout << "�ɹ��˳�" << std::endl;
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
