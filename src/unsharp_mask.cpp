#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <busin_log.h>
#include <string>
#include "test.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
/**
 * @brief 边缘增强算法 （效果与ImageJ中的Unsharp Mask 等价）
 * 通过从原图中减去图像的模糊版本（即非锐化蒙版）来锐化并增强边缘。
 * @param src -[in] 原图，类型可以为CV_8UC1/CV_16UC1/CV_8UC3
 * @param Radius -[in] 模糊半径（即被减去的高斯模糊的标准差Sigma），取值范围：大于0。单位：像素。增大高斯模糊半径会提高对比度
 * @param maskWeight -[in] 蒙版权重。取值范围:(0, 1).推荐范围:(0.1, 0.9)。决定滤镜的强度，蒙版权重=1 表示添加的高通滤波图像具有无限权重。提高蒙版权重值可进一步增强边缘效果。
 * @param dst -[in/out]
 */
void unsharp_mask(const cv::Mat& src, const double Radius, const double maskWeight, cv::Mat& dst) {
	businlog_error_assert_is_greater(Radius, 0, "");
	businlog_error_assert_is_greater(maskWeight, 0.0, "");
	businlog_error_assert_is_less(maskWeight, 1.0, "");
	businlog_error_assert(!src.empty(), "src is empty");
	// ---------- 类型处理策略 ----------
	// 为避免整数溢出，统一转到浮点空间运算，最后转回原类型
	// 8UC1/8UC3  -> 32FC1/32FC3
	// 16UC1      -> 32FC1
	cv::Mat srcFloat;
	int srcType = src.type();
	bool is8U = (srcType == CV_8UC1 || srcType == CV_8UC3);
	bool is16U = (srcType == CV_16UC1);

	if (is8U)
	{
		src.convertTo(srcFloat, CV_32F, 1.0 / 255.0);  // 归一化到 [0, 1]
	}
	else if (is16U)
	{
		src.convertTo(srcFloat, CV_32F, 1.0 / 65535.0); // 归一化到 [0, 1]
	}
	else
	{
		// 如果已经是浮点类型，直接克隆使用
		srcFloat = src.clone();
	}

	// ---------- Unsharp Mask 核心算法 ----------
	// 1. 高斯模糊（sigma = Radius）
	cv::Mat blurred;
	//GaussianBlur 传 Size(0,0)，OpenCV 会根据 sigma=Radius 自动计算核大小（6*sigma + 1 并取奇数）
	cv::GaussianBlur(srcFloat, blurred, cv::Size(0, 0), Radius);
	// 2. 计算高频细节 = 原图 - 模糊图
	cv::Mat highPass = srcFloat - blurred;

	// 3. 锐化结果 = 原图 + maskWeight * 高频细节
	//    等价于：dst = (1 - maskWeight) * blur + (1 + maskWeight/(1-maskWeight))...
	//    更直观的写法：
	cv::Mat sharpened = srcFloat + maskWeight * highPass;

	// ---------- 转回原类型 ----------
	if (is8U)
	{
		sharpened.convertTo(dst, srcType, 255.0);
	}
	else if (is16U)
	{
		sharpened.convertTo(dst, srcType, 65535.0);
	}
	else
	{
		sharpened.copyTo(dst);
	}
}
int test_unsharp_mask(std::string& err) {
	double Radius = 10;
	double MaskWeight = 0.7;
	std::string srcImgPath = "../test_imgs/wrinkle.png";
	const std::string dstDir = std::string(RESULT_IMAGES_DIR) + "/unsharpMask/";
	if (!std::filesystem::exists(dstDir))
	{
		std::error_code ec;
		std::filesystem::create_directories(dstDir, ec);
		businlog_error_return_err_reason(ec.value() == 0, "fail to create dir:" << dstDir, err, -1);
	}
	cv::Mat srcImg = cv::imread(srcImgPath, cv::IMREAD_UNCHANGED);
	businlog_error_return_err_reason(srcImg.empty() == false, "fail to imread from path:" << srcImgPath, err, -2);
	cv::Mat dstImg;
	unsharp_mask(srcImg, Radius, MaskWeight, dstImg);
	std::filesystem::path path_src(srcImgPath);
	auto dstPath = dstDir + "/" + path_src.stem().string() + "_sharped.png";
	businlog_error_return_err_reason(cv::imwrite(dstPath, dstImg), "fail to imwrite to path:" << dstPath, err, -3);
	return 0;
}