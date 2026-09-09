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

	cv::Mat srcFloat;
	int srcType = src.type();
	bool is8U = (srcType == CV_8UC1 || srcType == CV_8UC3);
	bool is16U = (srcType == CV_16UC1);

	// ========== 1、直接转float保留原始像素值 ==========
	if (is8U)
	{
		src.convertTo(srcFloat, CV_32F);
	}
	else if (is16U)
	{
		src.convertTo(srcFloat, CV_32F);
	}
	else
	{
		srcFloat = src.clone();
	}

	// ========== 2：高斯模糊边界模式改为 BORDER_REPLICATE，和ImageJ一致 ==========
	cv::Mat blurred;
	cv::GaussianBlur(srcFloat, blurred, cv::Size(0, 0), Radius, Radius, cv::BORDER_REPLICATE);

	// ========== 3、使用ImageJ原版公式 (src - w*blur)/(1-w) ==========
	cv::Mat sharpened;
	const double w = maskWeight;
	sharpened = (srcFloat - w * blurred) / (1.0 - w);

	// ========== 转回原类型，注意像素值截断（ImageJ内部会钳位到合法范围） ==========
	if (is8U)
	{
		// 限制 0~255，防止溢出
		cv::Mat clamped;
		cv::max(sharpened, 0.0f, clamped);
		cv::min(clamped, 255.0f, clamped);
		clamped.convertTo(dst, srcType);
	}
	else if (is16U)
	{
		// 16UC范围 0 ~ 65535
		cv::Mat clamped;
		cv::max(sharpened, 0.0f, clamped);
		cv::min(clamped, 65535.0, clamped);
		clamped.convertTo(dst, srcType);
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