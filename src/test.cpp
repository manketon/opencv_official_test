#include <opencv2/opencv.hpp>
#include <numeric>
#include <filesystem>
#include <vector>
#include "test.h"
#include "samples/cpp/common.h"
#include <fstream>
#include "hough.hpp"
using namespace cv;
using namespace std;
// 替换字符串中所有 from 为 to
std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
	size_t pos = 0;
	// 循环查找并替换，直到找不到目标子串
	while ((pos = str.find(from, pos)) != std::string::npos) {
		str.replace(pos, from.length(), to);
		// 避免重复替换刚替换的内容，更新查找起始位置
		pos += to.length();
	}
	return str;
}

//中心矩
double my_center_momnet(const std::vector<cv::Point>& pnts, const cv::Point2d& centroid, const int i, const int j)
{
	if (pnts.empty())
	{
		return 0.0;
	}
	double m = 0;
	for (const auto& pnt : pnts)
	{
		m += std::pow(centroid.y - pnt.y, i) * std::pow(centroid.x - pnt.x, j);
	}
	return m;
}

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
#if 1
	std::vector<cv::Point> pnts{ cv::Point(2, 2), cv::Point(2, 3), cv::Point(3, 2), /*cv::Point(3, 3),*/ };
#else
#endif
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

template <class T>
void find_different_pnts_impl(const cv::Mat& m1, const cv::Mat& m2)
{
	CV_Assert_4(m1.empty() == false, m2.empty() == false, m1.type() == m2.type(), m1.size == m2.size);
	for (int i = 0; i < m1.rows; ++i)
	{
		auto pM1_row = m1.ptr<T>(i);
		auto pM2_row = m2.ptr<T>(i);
		for (int j = 0; j < m1.cols; ++j)
		{
			if (pM1_row[j] != pM2_row[j])
			{
				std::cout << __FUNCTION__ << " | Not same pnt.x:" << j << ", pnt.y:" << i << std::endl;
			}
		}
	}
}
void find_different_pnts(const cv::Mat& m1, const cv::Mat& m2)
{
	const auto type = m1.type();
	switch (type)
	{
	case CV_8UC1:
		find_different_pnts_impl<uchar>(m1, m2);
		break;
	case CV_8UC3:
		find_different_pnts_impl<cv::Vec3b>(m1, m2);
		break;
	case CV_16UC1:
		find_different_pnts_impl<ushort>(m1, m2);
		break;
	default:
		std::cout << __FUNCTION__ << " | Not support type:" << cv::typeToString(type) << std::endl;
		CV_Assert(false);
	}
}
int test_find_different_pnts(std::string& str_err_reason)
{
	std::string str_first_img_path;
	std::cout << "请输入第一幅图路径:";
	std::cin >> str_first_img_path;
	std::string str_second_img_path;
	std::cout << "请输入第二幅图路径:";
	std::cin >> str_second_img_path;
	cv::Mat firstImg = cv::imread(str_first_img_path, cv::IMREAD_UNCHANGED);
	cv::Mat secondImg = cv::imread(str_second_img_path, cv::IMREAD_UNCHANGED);
	find_different_pnts(firstImg, secondImg);
	return 0;
}

// void calculateOrientationBasedPCA(const std::vector<cv::Point>& contour) {
// 	if (contour.size() < 5) return; // fitEllipse需要至少5个点  
// 
// 	// 使用PCA计算主方向  
// 	cv::PCA pca(contour, cv::Mat(), cv::PCA::DATA_AS_ROW);
// 	cv::Vec2f direction = pca.eigenvectors.at<float>(0); // 主成分方向  
// 
// 	// 计算角度
// 	float angle = atan2(direction[1], direction[0]) * 180.0 / CV_PI; // 将弧度转换为度  
// 	std::cout << "Principal Direction Angle: " << angle << std::endl;
// 
// 	// 可视化方向 (可选)  
// 	cv::Point2f mean(pca.mean.at<double>(0), pca.mean.at<double>(1));
// 	cv::Point2f endpoint(mean.x + direction[0] * 50, mean.y + direction[1] * 50);
// 	// 在图像上绘制  
// 	cv::Mat output;
// 	cv::cvtColor(inputImage, output, cv::COLOR_GRAY2BGR);
// 	cv::line(output, mean, endpoint, cv::Scalar(255, 0, 0), 2);
// 	cv::circle(output, mean, 5, cv::Scalar(0, 255, 0), -1);
// }


void calculateOrientationUsingPCA(const std::vector<cv::Point>& contour) {
	// 将轮廓点转为Mat格式，并转换为浮点数  
	cv::Mat data(contour.size(), 2, CV_32FC1);
	for (int i = 0; i < contour.size(); ++i)
	{
		auto pSrc_row = data.ptr<float>(i);
		pSrc_row[0] = contour[i].x;
		pSrc_row[1] = contour[i].y;
	}

	// 计算PCA  
	cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

	// 主成分方向  
	cv::Vec2f eigenvector = pca.eigenvectors.at<cv::Vec2f>(0);

	// 计算主方向的角度  
	double angle = std::atan2(eigenvector[1], eigenvector[0]) * 180.0 / CV_PI;

	// 轮廓中心  
	cv::Point2f center(pca.mean.at<double>(0), pca.mean.at<double>(1));

	std::cout << "中心: (" << center.x << ", " << center.y << "), 方向: " << angle << " degrees" << std::endl;
}

//基于轮廓矩来计算轮廓方向
void calculateOrientationBasedMoments(const std::vector<cv::Point>& contour) {
	// 计算轮廓的矩  
	cv::Moments m = cv::moments(contour, false);

	double cx = m.m10 / m.m00; // 轮廓的中心 x 坐标  
	double cy = m.m01 / m.m00; // 轮廓的中心 y 坐标  

	double mu20 = m.mu20; // 二阶矩  
	double mu11 = m.mu11; // 混合矩  
	double mu02 = m.mu02; // 二次矩  

	// 计算方向角  
	double theta = 0.5 * atan2(2 * mu11, mu20 - mu02); //atan2的返回值范围:(-PI, PI]

	// 方向以度为单位  
	double angle = theta * 180.0 / CV_PI;

	std::cout << "中心: (" << cx << ", " << cy << "), 方向: " << angle << " degrees" << std::endl;
}

int test_calc_direction_based_(std::string& str_err_reason)
{
	std::string str_src_bin_path;
	std::cout << "请输入二值图路径:";
	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);
	// 查找轮廓  
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(srcBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 遍历每个轮廓，计算并输出方向  
	for (const auto& contour : contours) {
		calculateOrientationBasedMoments(contour);
	}
	return 0;
}

void calc_moment_2(const std::vector<cv::Point>& contour)
{
	const auto m = cv::moments(contour, false);
	double area = m.m00; // 轮廓的面积
	 // 计算质心  
	Point2d center(m.m10 / area, m.m01 / area);

	// 计算主轴  
	double mu20 = m.m20 / area; // 归一化的矩  
	double mu02 = m.m02 / area; // 归一化的矩  
	double mu11 = m.m11 / area; // 归一化的矩  

	// 计算特征值  
	double nu20 = mu20 / (area * area);
	double nu02 = mu02 / (area * area);
	double nu11 = mu11 / (area * area);
	double area2 = cv::contourArea(contour);
}

void calc_moment_1(cv::InputArray _src)
{
	const auto m = cv::moments(_src, true);
	// 计算二阶矩  
	double mu20 = m.mu20;
	double mu11 = m.mu11;
	double mu02 = m.mu02;
	//TODO::目前长短轴和halcon::elliptic_axis还是有差别的，后面复现halcon::moments_region_2nd
	// 计算主方向和特征值  
	double theta = -0.5 * atan2(2 * mu11, mu20 - mu02);
	double length_a = sqrt(2 * (mu20 + mu02 + sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11))); //halcon中为8
	double length_b = sqrt(2 * (mu20 + mu02 - sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11))); //halcon中为8
}

void moments_region_2nd(const std::vector<cv::Point>& region, double& M11, double& M20, double& M02, double& Ia, double& Ib)
{
	if (region.empty())
	{
		M11 = M20 = M02 = Ia = Ib = 0;
	}
	const cv::Point2d centroid = area_center(region);
	M20 = my_center_momnet(region, centroid, 2, 0);
	M11 = my_center_momnet(region, centroid, 1, 1);
	M02 = my_center_momnet(region, centroid, 0, 2);
	const auto h = (M20 + M02) / 2;
	const auto delta = std::sqrt(h * h - M20 * M02 + M11 * M11);
	Ia = h + delta;
	Ib = h - delta;
}

void elliptic_axis(const std::vector<cv::Point>& region, double& Ra, double& Rb, double& Phi)
{
	if (region.size() <= 1)
	{
		Ra = Rb = Phi = 0.0;
		return;
	}
	double M11 = 0.0, M20 = 0.0, M02 = 0.0, Ia = 0.0, Ib = 0.0;
	moments_region_2nd(region, M11, M20, M02, Ia, Ib);
	const auto cof = region.size();
	//归一化
	M11 /= cof;
	M20 /= cof;
	M02 /= cof;
	//将归一化所得结果带入Halcon的elliptic_axis算子的公式，所得结果与elliptic_axis的结果一致
	Phi = -0.5 * atan2(2 * M11, M02 - M20);
	auto M20_add_M02 = M20 + M02;
	auto delta = std::sqrt((M20 - M02) * (M20 - M02) + 4 * M11 * M11);
	Ra = std::sqrt(8 * (M20_add_M02 + delta)) / 2;
	Rb = std::sqrt(8 * (M20_add_M02 - delta)) / 2;
}

void elliptic_axis_using_covMat(const std::vector<cv::Point>& region, double& Ra, double& Rb, double& Phi)
{
	if (region.size() <= 1)
	{
		Ra = Rb = Phi = 0.0;
		return;
	}
	const cv::Point2d centroid = area_center(region);
	// 构造协方差矩阵
	Mat covMat = Mat::zeros(2, 2, CV_64F);
	for (const auto& pt : region)
	{
		auto dx = static_cast<double>(pt.x) - centroid.x;
		auto dy = static_cast<double>(pt.y) - centroid.y;
		covMat.at<double>(0, 0) += dy * dy; //M20
		covMat.at<double>(0, 1) += dx * dy; //M11
		covMat.at<double>(1, 0) += dx * dy;
		covMat.at<double>(1, 1) += dx * dx; //M02
	}//所得矩阵与halcon::moments_region_2nd是一致的

	covMat /= region.size(); //归一化
	// 计算协方差矩阵的特征值和特征向量
	Point2f uv[2];
	Mat eigenVals, eigenVectors;
	cv::eigen(covMat, eigenVals, eigenVectors);
	uv[0] = Point2f(eigenVectors.at<double>(0, 0), eigenVectors.at<double>(0, 1));
	uv[1] = Point2f(eigenVectors.at<double>(1, 0), eigenVectors.at<double>(1, 1));
	//auto angleMinorAxis = atan2(uv[0].y, uv[0].x); //atan2的取值范围为(-PI, PI]
	//if (angleMinorAxis > CV_PI/2)
	//{
	//	angleMinorAxis -= CV_PI;
	//}
	//else if (angleMinorAxis < -CV_PI/2)
	//{
	//	angleMinorAxis += CV_PI;
	//}
	Phi = atan2(uv[1].y, uv[1].x); //atan2的取值范围为(-PI, PI]
	if (Phi > CV_PI/2)
	{
		Phi -= CV_PI;
	}
	else if (Phi < -CV_PI/2)
	{
		Phi += CV_PI;
	}
	//特征值为根为长短半轴的1/2
	double a = sqrt(eigenVals.at<double>(0, 0));
	double b = sqrt(eigenVals.at<double>(1, 0));
	Ra = max(a, b) * 2; //长半轴
	Rb = min(a, b) * 2; //短半轴
}

int test_elliptic_axis(std::string& str_err_reason)
{
	std::string str_src_bin_path = "D:/DevelopMent/LibLSR20_Optimized/testImg/only_one_region/one_region.png";
	std::cout << "请输入二值图路径:";
	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);

	std::vector<cv::Point> region;
	cv::findNonZero(srcBin, region);
	double Ra = 0.0, Rb = 0.0, theta = 0.0;
	elliptic_axis(region, Ra, Rb, theta);
	std::cout << "elliptic_axis result:" << std::endl;
	std::cout << "Ra=" << Ra << std::endl;
	std::cout << "Rb=" << Rb << std::endl;
	std::cout << "theta=" << theta << std::endl;
	Ra = 0.0, Rb = 0.0, theta = 0.0;
	elliptic_axis_using_covMat (region, Ra, Rb, theta);
	std::cout << "elliptic_axis_using_covMat result:" << std::endl;
	std::cout << "Ra=" << Ra << std::endl;
	std::cout << "Rb=" << Rb << std::endl;
	std::cout << "theta=" << theta << std::endl;
	return 0;
}


int test_mean_and_std_of_rows(std::string& str_err_reason)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入源图片目录地址:";
	std::cin >> str_src_imgs_dir;
	std::vector<std::string> vec_src_imgs_pathes;
	cv::glob(str_src_imgs_dir + "/*.png", vec_src_imgs_pathes);
	for (const std::string & str_src_img_path : vec_src_imgs_pathes)
	{
		cv::Mat img = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
		CV_Assert(img.empty() == false);
		cv::Mat meanImg(img.rows, 1, CV_64FC1);
		for (int i = 0; i < img.rows; ++i)
		{
			auto rowImg = img.row(i);
			meanImg.at<double>(i, 0) = cv::mean(rowImg)[0];
		}
		cv::Mat r, t;
		cv::meanStdDev(meanImg, r, t);
		auto meanValue = round(r.at<double>(0, 0), 2);
		auto stdDevValue = round(t.at<double>(0, 0), 2);
		std::cout << "image name:" << str_src_img_path << std::endl;
		std::cout << "           Mean:" << meanValue << std::endl;
		std::cout << "           StdDev:" << stdDevValue << std::endl;
		std::cout << "           Radio:" << stdDevValue * 100 / meanValue << std::endl;
	}
	return 0;
}

double round(double value, int n)
{
	double scale = std::pow(10.0, n); // 计算 10 的 n 次方  
	return std::round(value * scale) / scale; // 先乘以 scale 再四舍五入
}
static ushort calc_special_median(const cv::Mat& img16u)
{
	std::vector<ushort> vec_fit_values;
	std::vector<ushort> vec_all_values;
	vec_fit_values.reserve(img16u.cols * img16u.rows);
	vec_all_values.reserve(img16u.cols * img16u.rows);
	for (int i = 0; i < img16u.rows; ++i)
	{
		for (int j = 0; j < img16u.cols; ++j)
		{
			const auto gray = img16u.at<ushort>(i, j);
			vec_all_values.emplace_back(gray);
			if (gray > 10 && gray < 550)
			{
				vec_fit_values.emplace_back(gray);
			}
		}
	}
	if (vec_fit_values.empty())
	{
		double sum = std::accumulate(vec_all_values.begin(), vec_all_values.end(), 0.0);
		return static_cast<ushort>(sum/vec_all_values.size());
	}
	auto iter_median = vec_fit_values.begin() + ((vec_fit_values.size() - 1) >> 1);
	std::nth_element(vec_fit_values.begin(), iter_median, vec_fit_values.end());
	return *iter_median;
}
int test_HuaGeZi_and_calc_median(std::string& str_err_reason)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入源Polar图目录地址:";
	std::cin >> str_src_imgs_dir;
	std::string str_dst_imgs_dir;
	std::cout << "请输入存储目录地址:";
	std::cin >> str_dst_imgs_dir;
	std::vector<std::string> vec_src_polarImgs_pathes;
	cv::glob(str_src_imgs_dir + "/*.png", vec_src_polarImgs_pathes);
	int nStepSize = 4;
	double dTrimmingLength_mm = 3; //去边长度
	int nTrimmingLength_pixel = std::round(dTrimmingLength_mm * 1000 / nStepSize);
	int nGrid_width = std::round(2 * 1000 / nStepSize);
	int nGrid_height = nGrid_width;
	std::cout << "nTrimmingLength_pixel:" << nTrimmingLength_pixel << ", nGrid_width:" << nGrid_width << ", nGrid_height:" << nGrid_height << std::endl;
	for (const auto& str_src_img_path : vec_src_polarImgs_pathes)
	{
		cv::Mat srcPolarImg = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
		CV_Assert(srcPolarImg.empty() == false && srcPolarImg.type() == CV_16UC1);
		cv::Point center(srcPolarImg.cols / 2, srcPolarImg.rows / 2);
		cv::Mat polarImgRemovedEdge = srcPolarImg.clone();
		cv::circle(polarImgRemovedEdge, center, srcPolarImg.cols / 2, 0, 2 * nTrimmingLength_pixel);
		std::filesystem::path path_src(str_src_img_path);
		cv::imwrite(str_dst_imgs_dir + "/" + path_src.stem().string() + "_RemovedEdge.png", polarImgRemovedEdge);
		cv::Mat medianImg(polarImgRemovedEdge.rows/nGrid_height, polarImgRemovedEdge.cols / nGrid_width, CV_16UC1, cv::Scalar::all(0));
		for (int i = 0; i < medianImg.rows; ++i)
		{
			for (int j = 0; j < medianImg.cols; ++j)
			{
				cv::Rect roi(j * nGrid_width, i * nGrid_height, nGrid_width, nGrid_height);
				cv::Mat subImg(polarImgRemovedEdge, roi);
				auto medianV = calc_special_median(subImg);
				medianImg.at<ushort>(i, j) = medianV;
			}
		}
		cv::imwrite(str_dst_imgs_dir + "/" + path_src.stem().string() + "_median.png", medianImg);
	}
	return 0;
}

int test_getOnlyImgData()
{
	for (int i = 0; i < UINT_MAX; ++i)
	{
		const int imgWidth = 10000;
		const int imgHeight = imgWidth;
		uchar* pData = nullptr;
		{
			cv::Mat tmpImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar::all(255));
			pData = tmpImg.data; //获取字节流
			auto umat = tmpImg.u;
			tmpImg.data = nullptr;
			umat->origdata = nullptr;
		}
		cv::Mat img(imgHeight, imgWidth, CV_8UC1, pData); //使用上一个Mat的字节流重新构造一个Mat
		std::cout << "img.size:" << img.size() << std::endl;
		img = cv::Mat();//一定要先置空
		cv::fastFree(pData);
	}
	return 0;
}

int test_resize_images(std::string& str_err_reason)
{
	std::string str_src_imgs_dir;
	std::cout << "请输入源图片目录地址:";
	std::cin >> str_src_imgs_dir;

	std::string str_dst_imgs_dir = str_src_imgs_dir;
	std::cout << "请输入存储目录地址:";
	std::cin >> str_dst_imgs_dir;
	cv::Size dstSize(1500, 2000);
	std::vector < std::string > vec_src_imgs_pathes;
	cv::glob(str_src_imgs_dir + "/*", vec_src_imgs_pathes);
	for (const auto& str_src_img_path : vec_src_imgs_pathes)
	{
		cv::Mat srcImg = cv::imread(str_src_img_path, cv::IMREAD_UNCHANGED);
		CV_Assert(srcImg.empty() == false);
		if (srcImg.size() == dstSize)
		{
			continue;
		}
		cv::Mat resizedImg;
		cv::resize(srcImg, resizedImg, dstSize);
		std::filesystem::path path_src(str_src_img_path);
		std::string str_dst_path = str_dst_imgs_dir + "/" + path_src.stem().string() + "_W_" + std::to_string(dstSize.width)
			+ "_H_" + std::to_string(dstSize.height) + ".png";
		CV_Assert(cv::imwrite(str_dst_path, resizedImg));
		std::cout << __FUNCTION__ << " | Save resized image to path:" << str_dst_path << std::endl;
	}
	return 0;
}

int test_generate_compilation_script(std::string& str_err_reason)
{
	std::string str_src_dir;
	std::cout << "请输入源目录地址:";
	std::cin >> str_src_dir;
	//$(COMMON_LIB)/$(ProjectName)/$(Platform)/$(PlatformToolSet)/$(Configuration)/
	std::string str_dst_dir = "$(COMMON_LIB)/$(ProjectName)/$(Platform)/$(PlatformToolSet)/$(Configuration)";
	//str_dst_dir = "$(OutDir)";
	std::vector<std::string> vec_pathes;
	cv::glob(str_src_dir + "/*.*", vec_pathes, false);
	std::string str_dst_file_path = "./compilation_script.txt";
	std::ofstream fout(str_dst_file_path);
	CV_Assert(fout);
	fout << "setlocal" << std::endl;
	for (auto str_path : vec_pathes){
		const std::filesystem::path path_src(str_path); //使用$(Platform)代替x64，使用$(PlatformToolSet)代替v142，$(Configuration)代替Release/Debug
		if (path_src.extension() == ".lib") {
			//continue;
		}
		str_path = std::filesystem::absolute(str_path).string();
		fout << "D:/win_install/CMake/bin/cmake.exe -E copy_if_different ";
		str_path = replaceAll(str_path, "D:\\common_lib", "$(COMMON_LIB)");
		str_path = replaceAll(str_path, "x64", "$(Platform)");
		str_path = replaceAll(str_path, "v142", "$(PlatformToolSet)");
		str_path = replaceAll(str_path, "v143", "$(PlatformToolSet)");
		str_path = replaceAll(str_path, "Release", "$(Configuration)");
		str_path = replaceAll(str_path, "Debug", "$(Configuration)");
		std::string str_dst_path = str_dst_dir + "/" + path_src.filename().string();
		fout << str_path << " " << str_dst_path << std::endl;
		fout << "if %errorlevel% neq 0 goto :cmEnd" << std::endl;
	}
	fout << ":cmEnd" << std::endl;
	fout << "endlocal & call :cmErrorLevel %errorlevel% & goto :cmDone" << std::endl;
	fout << ":cmErrorLevel" << std::endl;
	fout << "exit /b %1" << std::endl;
	fout << ":cmDone" << std::endl;
	fout << "if %errorlevel% neq 0 goto :VCEnd" << std::endl;
	std::cout << "Save script to file:" << std::filesystem::absolute(str_dst_file_path).string() << std::endl;
	return 0;
}

int test_HoughLinesP(std::string& str_err_reason)
{
	int Votes_Lower_Limit = 40;
	double minLineLength = 80;
	double maxLineGap = 20;
	std::cout << "请输入二值图路径:";
	std::string str_bin_path;
	std::cin >> str_bin_path;
	cv::Mat srcBin = cv::imread(str_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);
	std::vector<cv::Vec4i> houghLines;
	//概率霍夫变换
	nsYRP::HoughLinesP(srcBin,	//输入图，可能会被修改
		houghLines,			//线条结果
		1,				//rho = 1
		CV_PI / 180,	//theta = 1弧度制	
		Votes_Lower_Limit,				//votes阈值
		minLineLength,				//直线最短长度
		maxLineGap);			//直线最大间隔
	std::cout << __FUNCTION__ << " | houghLines size:" << houghLines.size() << std::endl;
	cv::Mat stitchedBin = srcBin.clone();
	for (int i = 0; i < houghLines.size(); ++i)
	{
		const cv::Vec4i& vec4i = houghLines[i];
		const cv::Point begin(vec4i[0], vec4i[1]), end(vec4i[2], vec4i[3]);
		cv::line(stitchedBin, begin, end, 255, 1);
	}
	std::vector<cv::Mat> channels{srcBin, stitchedBin, srcBin};
	cv::Mat show_rslt;
	cv::merge(channels, show_rslt);
	std::filesystem::path path_src(str_bin_path);
	auto dst_path = path_src.parent_path().string() + "/" + path_src.stem().string() + "_stitched.png";
	cv::imwrite(dst_path, show_rslt);
	return 0;
}

int test_moment(std::string& str_err_reason)
{
	std::string str_src_bin_path = "D:/DevelopMent/LibLSR20_Optimized/testImg/only_one_region/one_region.png";
// 	std::cout << "请输入二值图路径:";
// 	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);
	calc_moment_1(srcBin);
	std::vector<cv::Point> region;
	cv::findNonZero(srcBin, region);
	calc_moment_2(region);
	// 查找轮廓  
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(srcBin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (size_t i = 0; i < contours.size(); i++)
	{
		calc_moment_2(contours[i]);
	}
	return 0;
}
