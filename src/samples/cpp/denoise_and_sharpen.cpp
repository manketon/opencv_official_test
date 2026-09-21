#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <busin_log.h>
#include <filesystem>
#include <opencv_tool/opencv_common_tool.h>
#include <opencv_tool/opencv_process_image.h>
#include "../../utils.hpp"

// 1) 频域平移：把零频分量移到图像中心
static void fftShift2D(cv::Mat& src)
{
	CV_Assert(src.channels() == 1 || src.channels() == 2);

	int cx = src.cols / 2;
	int cy = src.rows / 2;

	cv::Mat tmp;

	// 交换四个象限
	cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(src, cv::Rect(cx, 0, src.cols - cx, cy));
	cv::Mat q2(src, cv::Rect(0, cy, cx, src.rows - cy));
	cv::Mat q3(src, cv::Rect(cx, cy, src.cols - cx, src.rows - cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

/**
 * @brief 生成理想低通滤波器掩膜
 * @param imgSize -[in] 原图的尺寸
 * @param cutoff -[in] 截止频率，单位：像素
 * @return 掩膜图，尺寸为原图，值范围：[0, 1]，类型为 CV_32FC1。
 */
static cv::Mat ideal_lowpass_filter(const cv::Size& imgSize, const int cutoff)
{
	cv::Mat mask = cv::Mat::zeros(imgSize, CV_32F);
	const int rows = imgSize.height;
	const int cols = imgSize.width;
	int crow = rows / 2;
	int ccol = cols / 2;

	int x1 = std::max(0, ccol - cutoff);
	int x2 = std::min(cols, ccol + cutoff + 1);
	int y1 = std::max(0, crow - cutoff);
	int y2 = std::min(rows, crow + cutoff + 1);

	mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(1.0f);
	return mask;
}

/**
 * @brief 构建巴特沃斯低通滤波器掩模
 * @param rows -[in] 图高
 * @param cols -[in] 图宽
 * @param cutoff -[in] 截止频率（单位：像素），频谱中心到截止边缘的距离
 * @param n -[in] 滤波器阶数，控制截止边缘的陡峭程度。阶数越高，过渡边缘越陡峭，去噪效果越明显，但图像边缘的锐化效果越弱。Defaults to 2.
 * @return 掩膜图，尺寸为原图，值范围：[0, 1]，类型为 CV_32FC1。
 */
static cv::Mat butterworth_lowpass(const int rows, const int cols, const double cutoff, const int n = 2)
{
	cv::Mat mask(rows, cols, CV_32F);
	double cx = cols / 2.0;
	double cy = rows / 2.0;

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			double dx = x - cx;
			double dy = y - cy;
			double D = std::sqrt(dx * dx + dy * dy);
			mask.at<float>(y, x) = static_cast<float>(1.0 / (1.0 + std::pow(D / cutoff, 2.0 * n)));
		}
	}
	return mask;
}

// 4) 高斯低通滤波器
cv::Mat gaussian_lowpass(int rows, int cols, double sigma = 30.0)
{
	cv::Mat mask(rows, cols, CV_32F);
	double cx = cols / 2.0;
	double cy = rows / 2.0;

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			double dx = x - cx;
			double dy = y - cy;
			double D = std::sqrt(dx * dx + dy * dy);

			mask.at<float>(y, x) = static_cast<float>(std::exp(-(D * D) / (2.0 * sigma * sigma)));
		}
	}
	return mask;
}

// 5) 应用频域滤波器
cv::Mat apply_frequency_filter(const cv::Mat& src32F, const cv::Mat& mask)
{
	businlog_error_assert(src32F.depth() == CV_32F, "src32F.depth:" << cv::depthToString(src32F.depth()) << "  must be CV_32F");
	CV_Assert(src32F.size() == mask.size());

	cv::Mat img32F;
	if (src32F.depth() == CV_32F) {
		img32F = src32F;
	}
	else {
		int type = CV_MAKE_TYPE(CV_32F, src32F.channels());
		src32F.convertTo(img32F, type);
	}
	// 生成复数矩阵
	cv::Mat complexSrc;
	std::vector<cv::Mat> planes = {
		img32F.clone(),
		cv::Mat::zeros(img32F.size(), CV_32F)
	};
	cv::merge(planes, complexSrc);

	// FFT
	cv::dft(complexSrc, complexSrc);
	fftShift2D(complexSrc);

	// 乘以掩模
	std::vector<cv::Mat> fplanes(2);
	cv::split(complexSrc, fplanes);
	cv::Mat maskF = mask.clone();
	fplanes[0] = fplanes[0].mul(maskF);
	fplanes[1] = fplanes[1].mul(maskF);
	cv::merge(fplanes, complexSrc);

	// 逆 FFT
	fftShift2D(complexSrc); // 对应 Python 中 ifftshift
	cv::dft(complexSrc, complexSrc, cv::DftFlags::DFT_INVERSE + cv::DftFlags::DFT_SCALE);

	// 取幅度：np.abs(ifft2(...))
	cv::split(complexSrc, fplanes);
	complexSrc.release();
	cv::Mat mag;
	cv::magnitude(fplanes[0], fplanes[1], mag);
	return mag;
}

// 6) 高提升滤波（等价于 Python 中的 high_boost_filter）
cv::Mat high_boost_filter(const cv::Mat& image,
	const double cutoff,
	const double boost_factor = 1.5,
	const std::string& filter_type = "gaussian")
{
	cv::Mat lp_mask;
	if (filter_type == "gaussian")
	{
		lp_mask = gaussian_lowpass(image.rows, image.cols, cutoff);
	}
	else if (filter_type == "butterworth")
	{
		lp_mask = butterworth_lowpass(image.rows, image.cols, cutoff, 2);
	}
	else // ideal
	{
		lp_mask = ideal_lowpass_filter(image.size(), (int)cutoff);
	}

	// 高通 = 1 - 低通
	cv::Mat hp_mask = cv::Mat::ones(image.size(), CV_32F) - lp_mask;

	// 高提升掩模 = 1 + boost_factor * hp_mask
	cv::Mat boost_mask = cv::Mat::ones(image.size(), CV_32F) + boost_factor * hp_mask;

	// 频域增强
	auto dst = apply_frequency_filter(image, boost_mask);
	return dst;
}
/**
 * @brief
 * @param src -[in/out]
 * @param sharpened -[out] 降噪和锐化后的图像，类型和尺寸和输入一致。
 * @param sigma -[in/out]
 * @param cutoff -[in/out]
 * @param boost_factor -[in/out]
 */
void denoise_and_sharpen(const cv::Mat& src, cv::Mat& sharpened, const double sigma = 71,
	const double cutoff = 25.0, const double boost_factor = 1.7) {
	// 1) 生成高斯低通掩模
	cv::Mat gmask = gaussian_lowpass(src.rows, src.cols, sigma);
	// 2) 频域低通滤波
	cv::Mat src32F;
	if (src.depth() == CV_32F) {
		src32F = src;
	}
	else {
		int type = CV_MAKE_TYPE(CV_32F, src.channels());
		src.convertTo(src32F, type);
	}
	cv::Mat denoised = apply_frequency_filter(src32F, gmask);
	gmask.release();
	// 3) 高频增强锐化
	sharpened = high_boost_filter(denoised, cutoff, boost_factor, "butterworth");
	if (sharpened.depth() != src.depth()) {
		int type = CV_MAKE_TYPE(src.depth(), sharpened.channels());
		sharpened.convertTo(sharpened, type);
	}
}

void denoise_and_sharpen_by_slices(const cv::Mat& src, cv::Mat& sharpened, const double sigma = 71,
	const double cutoff = 25.0, const double boost_factor = 1.7) {
	cv::Size sliceSize = cv::Size(std::min(/*2048*/1024, src.cols), std::min(1024, src.rows)); //由于opencv不支持大图，故需要切片，这些参数不需要暴露出去。
	const int overlapLength = std::max(static_cast<int>(cutoff), static_cast<int>(sigma)); //TODO::即使取大点，还是存在切割线情况。
	std::vector <nsTest::ImgSliceRslt > slicedRslts;
	std::string err;
	int ret = nsTest::sliceImage(src, sliceSize.width, sliceSize.height, overlapLength, overlapLength, slicedRslts, err);
	businlog_error_assert(!ret, "Fail to slice image, reason:" << err);
	businlog_info("slicedRslts.size:" << slicedRslts.size());
	sharpened.create(src.size(), src.type());
	for (const auto& sliceRslt : slicedRslts) {
		cv::Mat localRslt;
		denoise_and_sharpen(sliceRslt.subImg, localRslt, sigma, cutoff, boost_factor);
		cv::Mat localSharpened(sharpened, sliceRslt.roi);
		localRslt.copyTo(localSharpened);
	}
}
int test_denoise_and_sharpen(std::string& err)
{
	std::string img_path = R"(D:\DevelopMent\LibLSR20_Optimized\doc\CutOffFilter\srcImgs\20260902_090032\gray_Gaussian_Long WavelengthDUIBIAO Wrinkle 1A_SJ DUIBIAO wrinkle 1A_01_1_SZcO_IPP.png)";
	//img_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\srcImgs\3.png)";
	cv::Mat src = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	businlog_error_return_err_reason(!src.empty(), "fail to imread image from path:" << img_path, err, -1);
	cv::Mat dst;
	double sigma = 71;
	double cutoff = 25.0;
	double boost_factor = 1.7;
	std::ostringstream oss;
	oss << sigma << "_" << cutoff << "_" << boost_factor;
	std::string params = oss.str();
	denoise_and_sharpen_by_slices(src, dst, sigma, cutoff, boost_factor);
	std::filesystem::path path_src(img_path);
	std::string out_path = path_src.parent_path().string() + "/" + path_src.stem().string() + "_" + params + "_sharpened.png";
	businlog_error_return_err_reason(cv::imwrite(out_path, dst), "fail to write image to path:" << out_path, err, -1);
	businlog_info("success to write image to path:" << out_path);
	return 0;
}