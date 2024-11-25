#include <opencv2/opencv.hpp>
#include <numeric>
#include <filesystem>
#include <vector>
#include "test.h"
using namespace cv;
using namespace std;
int test_load_img_from_bytes()
{
	std::string str_src_img_path = "D:/wafer_images/Si PSL 83nm_SSc/Si PSL 83nm_SSc";
	int nImg_width = 32768;
	int nImg_height = 6001;
	//! 以二进制流方式读取图片到内存
	FILE* pFile = fopen(str_src_img_path.c_str(), "rb");
	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);
	CV_Assert(lSize % 2 == 0);
	rewind(pFile);
	const long pixel_num = lSize / 2;
	CV_Assert(pixel_num == nImg_height * nImg_width);
	ushort* pData = new ushort[pixel_num];
	fread(pData, sizeof(ushort), pixel_num, pFile);
	fclose(pFile);
	cv::Mat img(nImg_height, nImg_width, CV_16UC1);
	for (int i = 0; i < nImg_height; ++i)
	{
		for (int j = 0; j < nImg_width; ++j)
		{
			img.at<ushort>(i, j) = pData[i * nImg_width + j];
		}
	}
	delete[] pData;
	cv::imwrite("D:/wafer_images/Si PSL 83nm_SSc/rslt.png", img);
	return 0;
}

struct SInfo
{
	SInfo()
	{

	}
	double meanV = 0.0;
	double medianV = 0.0;
	double stdDevFromMean = 0.0;
	double stdDevFromMedian = 0.0;
};
std::ostream& operator<<(std::ostream& outPutStream, SInfo& info)
{
	outPutStream << "meanV:" << info.meanV << ", stdDevFromMean:" << info.stdDevFromMean << ", medianV:" << info.medianV << ", stdDevFromMedian:"
		<< info.stdDevFromMedian
		;
	return outPutStream;
}

std::string get_channel_name(const std::string& str_img_path)
{
	//240625 - 1_650V_33_V1_4N46 - FP02015_25_NUV - PL_IPP.png
	auto last_index = str_img_path.rfind('_');
	CV_Assert(last_index != std::string::npos);
	auto second_index = str_img_path.rfind('_', last_index);
	CV_Assert(second_index != std::string::npos);
	--second_index;
	return str_img_path.substr(second_index, last_index - second_index);
}

template<class T>
SInfo get_info(const T* pFirst, const T* pLast)
{
	SInfo info;
	const auto n = pLast - pFirst;
	CV_Assert(n > 0);
 	auto iter_median = pFirst + (n - 1) / 2;
	info.medianV = *iter_median;
	info.meanV = std::accumulate(pFirst, pLast, 0.0) / n;
	for (auto p = pFirst; p < pLast; ++p)
	{
		const auto value = *p;
		info.stdDevFromMean += std::pow(value - info.meanV, 2);
		info.stdDevFromMedian += std::pow(value - info.medianV, 2);
	}
	info.stdDevFromMean = std::sqrt(info.stdDevFromMean / n);
	info.stdDevFromMedian = std::sqrt(info.stdDevFromMedian / n);
	return info;
}

int show_img_info(const cv::Mat& img, const std::string& str_src_img_path, std::string& str_err_reason)
{
	const int roi_height = 1000;
	std::filesystem::path path_src(str_src_img_path);
	cv::Rect roi(0, 0, img.cols, roi_height);
	std::cout << "image:" << path_src.filename().string() << std::endl;
	for (roi.y = 300; roi.y < img.rows; roi.y += roi.height)
	{
		if (roi.y + roi.height - 1 >= img.rows)
		{
			roi.height = img.rows - roi.y;
		}
		std::cout << "\t行区间[" << roi.y << ":" << roi.y + roi.height << "]" << std::endl;
		cv::Mat subImg(img, roi);
		const size_t imgLength = subImg.rows * subImg.cols;
		const ushort* pSrcImgData = subImg.ptr<ushort>(0);
		for (double cutoff = 0.01; cutoff < 0.5; cutoff += 0.03)
		{
			std::vector<ushort> tmp_vec(pSrcImgData, pSrcImgData + imgLength);
			std::sort(tmp_vec.begin(), tmp_vec.end());
			auto pFirst = tmp_vec.data() + static_cast<size_t>(imgLength * cutoff);
			auto pLast = tmp_vec.data() + static_cast<size_t>(imgLength * (1.0 - cutoff)) + 1;
			auto info = get_info<ushort>(pFirst, pLast);
			std::cout << "\t\t区间[" << cutoff * 100 << "%:" << 100 * (1 - cutoff) << "%]的数据分布:" << info << std::endl;
		}
	}
	
	return 0;
}

int test_imgs_statistical_informations(std::string& str_err_reason)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入原图片目录地址:";
	std::cin >> str_src_imgs_dir;
	std::vector<std::string> vec_src_imgs_pathes;
	cv::glob(str_src_imgs_dir + "/*.png", vec_src_imgs_pathes);

// 	std::string str_bin_imgs_dir;
// 	std::cout << "请输入二值图目录地址:";
// 	std::cin >> str_bin_imgs_dir;

// 	std::map<std::string, std::string> map_srcImgName_binImgName;
// 	map_srcImgName_binImgName.insert({"240625-1_650V_33_V1_4N46-FP02015_25_NUV-PL_IPP.png", "4N46-FP02015_NUV-PL_Defect_Binary_SD5.png"});
// 	map_srcImgName_binImgName.insert({"240625-1_650V_33_V1_4N46-FP02015_25_QScO_IPP.png", "4N46-FP02015_QScO_Defect_Binary_SD2.png"});
// 	map_srcImgName_binImgName.insert({ "240625-1_650V_33_V1_4N46-FP02015_25_QZrO_IPP.png", "4N46-FP02015_QZrO_Defect_Binary_SD3.png" });
// 	map_srcImgName_binImgName.insert({ "240625-1_650V_33_V1_4N46-FP02015_25_ScN_IPP.png", "4N46-FP02015_ScN_Defect_Binary_SD1.png" });
// 	map_srcImgName_binImgName.insert({ "240625-1_650V_33_V1_4N46-FP02015_25_VIS-PL_IPP.png", "4N46-FP02015_VIS-PL_Defect_Binary_SD4.png" });
	for (const auto& str_src_img_path : vec_src_imgs_pathes)
	{
		const cv::Mat srcImg = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
		CV_Assert(srcImg.empty() == false);
		CV_Assert(srcImg.type() == CV_16UC1);
		int ret = show_img_info(srcImg, str_src_img_path, str_err_reason);
		if (ret)
		{
			std::cout << __FUNCTION__ << " | error, ret:" << ret << ", img path:" << str_src_img_path << std::endl;
			return ret;
		}
// 		auto iter = map_srcImgName_binImgName.find(std::filesystem::path(str_src_img_path).filename().string());
// 		CV_Assert(iter != map_srcImgName_binImgName.end());
// 		auto str_bin_img_path = str_bin_imgs_dir + "/" + iter->second;
// 		cv::Mat binImg = cv::imread(str_bin_img_path, cv::IMREAD_UNCHANGED);
// 		CV_Assert(binImg.empty() == false);
// 		binImg = binImg > 0;
	}
	return 0;
}
cv::Mat towRows2oneRow(const cv::Mat& src)
{
	int nDstWidth = src.cols * 2;
	int nDstHeight = src.rows / 2;
	cv::Mat dst(nDstHeight, nDstWidth, src.type());
	for (int i = 0; i < nDstHeight; ++i)
	{
		cv::Rect leftROI(0, i, src.cols, 1);
		cv::Mat leftSubImg(dst, leftROI);
		src.row(2 * i).copyTo(leftSubImg);
		cv::Rect rightROI = leftROI;
		rightROI.x = src.cols;
		cv::Mat rightSubImg(dst, rightROI);
		src.row(2 * i + 1).copyTo(rightSubImg);
	}
	return dst;
}
int test_towRows2oneRow(std::string& str_err_reason)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入原图片目录地址:";
	std::cin >> str_src_imgs_dir;
	std::vector<std::string> vec_src_imgs_pathes;
	cv::glob(str_src_imgs_dir + "/*.png", vec_src_imgs_pathes);
	std::string str_dst_imgs_dir;
	std::cout << "请输入存储目录地址:";
	std::cin >> str_dst_imgs_dir;
	for (const auto& str_src_img_path : vec_src_imgs_pathes)
	{
		if (str_src_img_path.find("SC") == std::string::npos)
		{
			continue;
		}
		const cv::Mat srcImg = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
		CV_Assert(srcImg.empty() == false);
		CV_Assert(srcImg.type() == CV_16UC1);
		cv::Mat dstImg = towRows2oneRow(srcImg);
		//if (dstImg.rows == 500)
		//{
		//	cv::copyMakeBorder(dstImg, dstImg, 0, 1, 0, 0, cv::BORDER_DEFAULT);
		//}
		//else if (dstImg.rows == 502)
		//{
		//	dstImg = cv::Mat(dstImg, cv::Rect(0, 0, dstImg.cols, 501));
		//}
		std::string str_dst_img_path = str_dst_imgs_dir + "/" + std::filesystem::path(str_src_img_path).filename().string();
		CV_Assert(cv::imwrite(str_dst_img_path, dstImg));
	}
	return 0;
}



int test_minAreaRect(std::string& str_err_reason)
{
	std::vector<cv::Point> pnts{ cv::Point(2, 2), cv::Point(3, 3)/*, cv::Point(4, 5), cv::Point(5, 6), cv::Point(5, 5)*/ };
	auto rect = cv::minAreaRect(pnts);
	std::cout << __FUNCTION__ << " | size:" << rect.size << ", angle:"  << rect.angle << ", center:" << rect.center << std::endl;
	std::vector<cv::Point2f> corner_pnts(4);
	rect.points(corner_pnts.data());
	for (auto& corner_pnt : corner_pnts)
	{
		std::cout << "corner_pnt:" << corner_pnt << ", ";
	}
	std::cout << std::endl;
	return 0;
}

int test_convexhull(std::string& str_err_reason)
{
	std::string str_src_bin_path = "../test_imgs/polarBin.png";
	std::cout << "请输入源二值图路径:";
	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<int>> hull(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hull[i], false);
	}

	std::vector<std::vector<cv::Vec4i>> defects(contours.size()); 
	for (int i = 0; i < contours.size(); i++) {
		if (hull[i].size() > 3) {
			convexityDefects(contours[i], hull[i], defects[i]);
		}
	}
	cv::Mat show_rslt;
	cv::cvtColor(srcBin, show_rslt, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < contours.size(); i++) {
		for (int j = 0; j < defects[i].size(); j++) {
			cv::Vec4i& v = defects[i][j];
			float depth = v[3] / 256.0;
			if (depth > 10) {
				int startidx = v[0];
				cv::Point start(contours[i][startidx]);
				int endidx = v[1];
				cv::Point end(contours[i][endidx]);
				int faridx = v[2];
				cv::Point far(contours[i][faridx]);
				cv::line(show_rslt, start, end, cv::Scalar(0, 0, 255), 2);
				cv::line(show_rslt, start, far, cv::Scalar(0, 255, 0), 2);
				cv::line(show_rslt, end, far, cv::Scalar(0, 255, 0), 2);
				cv::circle(show_rslt, far, 4, cv::Scalar(0, 255, 0), -1);
			}
		}
	}
	std::string str_dst_img_path = std::string(RESULT_IMAGES_DIR) + "/bin_with_convexHull.png";
	CV_Assert(cv::imwrite(str_dst_img_path, show_rslt));
	return 0;
}

int test_HoughLines(std::string& str_err_reason)
{
	std::string str_src_bin_path = "../test_imgs/polarBin.png";
	std::cout << "请输入源二值图路径:";
	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);
	std::filesystem::path path_src(str_src_bin_path);
	//霍夫变换（注意：霍夫变换可能会改变源二值图）
	vector<Vec2f> lines;
	HoughLines(srcBin,	//输入图
		lines,			//线条结果
		1,				//rho = 1
		CV_PI / 180,	//theta = 1弧度制
		150,			//votes阈值
		0,				//srn默认为0
		0);				//stn默认为0

	Mat dst(srcBin.size(), CV_8UC1, cv::Scalar::all(0));
	//显示检测到的线条为红色
	for (size_t i{ 0 }; i < lines.size(); i++) {
		float rho{ lines[i][0] }, theta{ lines[i][1] };		//分别获取rho和theta值
		Point pt1, pt2;
		double a{ cos(theta) }, b{ sin(theta) };
		double x0 = a * rho, y0 = b * rho;					//通过rho和theta计算极坐标向量的端点
		//计算直线上两点的坐标
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dst, pt1, pt2, Scalar::all(255), 3, LINE_8/*LINE_AA*/);
	}
	{
		std::vector<cv::Mat> vec_channels{ srcBin, dst, srcBin };
		cv::Mat result;
		cv::merge(vec_channels, result);
		std::string str_dst_path = std::string(RESULT_IMAGES_DIR) + "/" + path_src.stem().string() + "_houghLines.png";
		CV_Assert(cv::imwrite(str_dst_path, result));
	}

	cv::Mat dstP(srcBin.size(), CV_8UC1, cv::Scalar::all(0));
	//概率霍夫变换
	vector<Vec4i> linesP;
	HoughLinesP(srcBin,	//输入图
		linesP,			//线条结果
		1,				//rho = 1
		CV_PI / 180,	//theta = 1弧度制	
		50,				//votes阈值
		50,				//直线最短长度
		20);			//直线最大间隔
	//显示结果线条
	for (size_t i{ 0 }; i < linesP.size(); i++) {
		Vec4i l{ linesP[i] };
		line(dstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar::all(255), 1, LINE_8);
	}

	{
		std::vector<cv::Mat> vec_channels{ srcBin, dstP, srcBin };
		cv::Mat rsult_P;
		cv::merge(vec_channels, rsult_P);
		std::string str_dstP_path = std::string(RESULT_IMAGES_DIR) + "/" + path_src.stem().string() + "_houghLinesP.png";
		CV_Assert(cv::imwrite(str_dstP_path, rsult_P));
	}
	return 0;
}
