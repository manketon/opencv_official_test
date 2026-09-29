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

// numpy.percentile(x, (p0, p1), method="linear") 返回中心值
static double percentile_of_mat(const cv::Mat& m, double p) {
	CV_Assert(m.total() > 0);
	size_t n = static_cast<size_t>(m.total());
	if (p <= 0.0) return m.ptr<double>()[0];   // 我们在外层保证 p ∈ (0,100)
	size_t idx = static_cast<size_t>(static_cast<double>(n - 1) * (p / 100.0));
	// 取第 idx 小（linear 法）：用 nth_element（平均 O(n)）
	std::vector<double> v(m.begin<double>(), m.end<double>());
	std::nth_element(v.begin(), v.begin() + idx, v.end());
	return v[idx];
}
// 线性插值版（完全跟 numpy.percentile 的线性插值一致）
static double percentile_of_mat_linear(const cv::Mat& m, double p) {
	CV_Assert(m.total() > 0);
	size_t n = static_cast<size_t>(m.total());
	double h = (n - 1) * (p / 100.0);
	size_t lo = static_cast<size_t>(std::floor(h));
	size_t hi = static_cast<size_t>(std::ceil(h));
	double frac = h - lo;
	std::vector<double> v(m.begin<double>(), m.end<double>());
	auto begin = v.begin();
	if (lo == hi) {
		std::nth_element(begin, begin + lo, v.end());
		return v[lo];
	}
	else {
		std::nth_element(begin, begin + lo, v.end());            // lo
		auto next = begin + lo + 1;
		std::nth_element(next, v.end(), v.end());                // 在 [lo+1, end) 内找 min，即原 hi
		// 简化：再用 nth_element 在 lo+1 处
		std::nth_element(next, next, v.end());                   // 此时 *next 是 [lo+1, end) 的最小，就是 hi
		return v[lo] + frac * (v[lo + 1] - v[lo]);
	}
}
/**
 * 把浮点图像按 P0.1/P99.9 百分位拉伸到 [0,255]，再转 uint8。
 * 与 Python:
 *   lo, hi = np.percentile(arr, (0.1, 99.9))
 *   if hi <= lo: hi = lo + 1.0
 *   out = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
 * 完全等价。
 *
 * 支持 CV_32F / CV_64F，任意通道数；内部转连续 CV_64F 计算。
 */
static void to_uint8(const cv::Mat& src, cv::Mat& dst,
	double start_ = 0.1, double end_ = 99.9) {
	CV_Assert(src.type() == CV_32F || src.type() == CV_64F);
	CV_Assert(start_ >= 0.0 && start_ < end_&& end_ <= 100.0);
	CV_Assert(src.total() > 0);
	cv::Mat f64;
	src.convertTo(f64, CV_64F, 1.0, 0.0);   // 统一走 double，单次 transposed 不增负担
	// 1) 取两个分位数
	double lo = percentile_of_mat(f64, start_);
	double hi = percentile_of_mat(f64, end_);
	if (hi <= lo) hi = lo + 1.0;
	// 2) 逐像素: out = clip((x - lo) * 255.0 / (hi - lo), 0, 255)
	//   用 convertTo 的 scale/shift 一步搞定，无需手写逐像素循环
	const double k = 255.0 / (hi - lo);
	const double b = -k * lo;
	cv::Mat f32;
	f64.convertTo(f32, CV_32F, k, b);            // x = k*src + b
	f32.convertTo(dst, CV_8U, 1.0, 0.0);         // 自动 clip 到 [0,255]
}

// 1) 频域平移：严格等价于 NumPy 的 fftshift / ifftshift
//    对奇数尺寸，必须使用 n // 2，而不是 (n+1)//2 或半像素中心。
static void fftShift2D(cv::Mat& src)
{
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	if (src.empty()) return;

	cv::Mat dst = src.clone();
	const int rowShift = src.rows / 2;
	const int colShift = src.cols / 2;

	if (src.channels() == 1)
	{
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				const int sy = (y + rowShift) % src.rows;
				const int sx = (x + colShift) % src.cols;
				dst.at<double>(y, x) = src.at<double>(sy, sx);
			}
		}
	}
	else
	{
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				const int sy = (y + rowShift) % src.rows;
				const int sx = (x + colShift) % src.cols;
				dst.at<cv::Vec2d>(y, x) = src.at<cv::Vec2d>(sy, sx);
			}
		}
	}

	src = dst;
}

static void ifftShift2D(cv::Mat& src)
{
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	if (src.empty()) return;

	cv::Mat dst = src.clone();
	const int rowShift = -(src.rows / 2);
	const int colShift = -(src.cols / 2);

	if (src.channels() == 1)
	{
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				const int sy = (y + rowShift + src.rows) % src.rows;
				const int sx = (x + colShift + src.cols) % src.cols;
				dst.at<double>(y, x) = src.at<double>(sy, sx);
			}
		}
	}
	else
	{
		for (int y = 0; y < src.rows; ++y)
		{
			for (int x = 0; x < src.cols; ++x)
			{
				const int sy = (y + rowShift + src.rows) % src.rows;
				const int sx = (x + colShift + src.cols) % src.cols;
				dst.at<cv::Vec2d>(y, x) = src.at<cv::Vec2d>(sy, sx);
			}
		}
	}

	src = dst;
}

/**
 * @brief 生成理想低通滤波器掩膜
 * @param imgSize -[in] 原图的尺寸
 * @param cutoff -[in] 截止频率，单位：像素
 * @return 掩膜图，尺寸为原图，值范围：[0, 1]，类型为 CV_64FC1。
 */
static cv::Mat ideal_lowpass_filter(const cv::Size& imgSize, const int cutoff)
{
	cv::Mat mask = cv::Mat::zeros(imgSize, CV_64F);
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
 * @return 掩膜图，尺寸为原图，值范围：[0, 1]，类型为 CV_64FC1。
 */
static cv::Mat butterworth_lowpass(const int rows, const int cols, const double cutoff, const int n = 2)
{
	CV_Assert(cutoff > 0.0);
	cv::Mat mask(rows, cols, CV_64F);
	const int cx = cols / 2;
	const int cy = rows / 2;

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			double dx = static_cast<double>(x - cx);
			double dy = static_cast<double>(y - cy);
			double D = std::sqrt(dx * dx + dy * dy);
			mask.at<double>(y, x) = 1.0 / (1.0 + std::pow(D / cutoff, 2.0 * n));
		}
	}
	return mask;
}

// 4) 高斯低通滤波器
cv::Mat gaussian_lowpass(int rows, int cols, double sigma = 30.0)
{
	CV_Assert(sigma > 0.0);
	cv::Mat mask(rows, cols, CV_64F);
	const int cx = cols / 2;
	const int cy = rows / 2;
	const double denom = 2.0 * sigma * sigma;

	for (int y = 0; y < rows; ++y)
	{
		double* p = mask.ptr<double>(y);
		for (int x = 0; x < cols; ++x)
		{
			double dx = static_cast<double>(x - cx);
			double dy = static_cast<double>(y - cy);
			double r_sq = dx * dx + dy * dy;
			p[x] = std::exp(-r_sq / denom);
		}
	}
	return mask;
}

// 5) 应用频域滤波器：等价 Python 中 np.fft.ifft2(np.fft.ifftshift(fshift)).real
cv::Mat apply_frequency_filter(const cv::Mat& src32F, const cv::Mat& mask)
{
	CV_Assert(src32F.size() == mask.size());
	CV_Assert(src32F.channels() == 1 || src32F.channels() == mask.channels());

	cv::Mat img64F;
	if (src32F.depth() == CV_64F) {
		img64F = src32F;
	}
	else {
		int type = CV_MAKE_TYPE(CV_64F, src32F.channels());
		src32F.convertTo(img64F, type);
	}

	cv::Mat complexSrc;
	std::vector<cv::Mat> planes = {
		img64F,
		cv::Mat::zeros(img64F.size(), CV_64F)
	};
	cv::merge(planes, complexSrc);
	planes.clear();

	cv::dft(complexSrc, complexSrc);
	fftShift2D(complexSrc);

	std::vector<cv::Mat> fplanes(2);
	cv::split(complexSrc, fplanes);
	cv::Mat maskF = mask.clone();
	if (maskF.depth() != CV_64F) {
		maskF.convertTo(maskF, CV_64F);
	}
	fplanes[0] = fplanes[0].mul(maskF);
	fplanes[1] = fplanes[1].mul(maskF);
	cv::merge(fplanes, complexSrc);
	maskF.release();
	ifftShift2D(complexSrc);
	cv::dft(complexSrc, complexSrc, cv::DftFlags::DFT_INVERSE | cv::DftFlags::DFT_SCALE);

	cv::split(complexSrc, fplanes);
	cv::Mat realPart = fplanes[0].clone();
	return realPart;
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
		lp_mask = ideal_lowpass_filter(image.size(), static_cast<int>(cutoff));
	}

	cv::Mat hp_mask = cv::Mat::ones(image.size(), CV_64F) - lp_mask;
	lp_mask.release();
	cv::Mat boost_mask = cv::Mat::ones(image.size(), CV_64F) + boost_factor * hp_mask;
	hp_mask.release();
	auto dst = apply_frequency_filter(image, boost_mask);
	return dst;
}
static std::pair<double, double> _scale_params(const int rows, const int cols,
	const double base_short_axis = 468.0,
	const double sigma = 71.0,
	const double cutoff = 25.0)
{
	const double short_axis = static_cast<double>(std::min(rows, cols));
	const double scale = short_axis / base_short_axis;
	return { sigma * scale, cutoff * scale };
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
	CV_Assert(src.channels() == 1 || src.channels() == 3 || src.channels() == 4);
	const auto [sigma_s, cutoff_s] = _scale_params(src.rows, src.cols, 468.0, sigma, cutoff);
	businlog_info("srcImg.size:" << src.size() << ", sigma:" << sigma << ", sigma_s:" << sigma_s << ", cutoff:" << cutoff
		<< ", cutoff_s:" << cutoff_s);
	cv::Mat src64F;
	if (src.depth() == CV_64F) {
		src64F = src;
	}
	else {
		int type = CV_MAKE_TYPE(CV_64F, src.channels());
		src.convertTo(src64F, type);
	}

	cv::Mat gmask = gaussian_lowpass(src.rows, src.cols, sigma_s);
	cv::Mat denoised = apply_frequency_filter(src64F, gmask);
	gmask.release();
	src64F.release();
	cv::Mat sharpened64 = high_boost_filter(denoised, cutoff_s, boost_factor, "butterworth");
	cv::Mat tmp;
	to_uint8(sharpened64, tmp, 0.1, 99.9);
	sharpened = tmp;
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
	//img_path = R"(F:\workspace\WaferInspection\LibLSR20_Optimized\testImg\unsharp_mask\wrinkle_sharped.png)";
	cv::Mat src = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	businlog_error_return_err_reason(!src.empty(), "fail to imread image from path:" << img_path, err, -1);
	cv::Mat dst;
	double sigma = 71;
	double cutoff = 25.0;
	double boost_factor = 1.7;
	std::ostringstream oss;
	oss << sigma << "_" << cutoff << "_" << boost_factor;
	std::string params = oss.str();
	denoise_and_sharpen(src, dst, sigma, cutoff, boost_factor);
	std::filesystem::path path_src(img_path);
	std::string out_path = path_src.parent_path().string() + "/" + path_src.stem().string() + "_" + params + "_sharpened.png";
	businlog_error_return_err_reason(cv::imwrite(out_path, dst), "fail to write image to path:" << out_path, err, -1);
	businlog_info("success to write image to path:" << out_path);
	return 0;
}
//测试ShitFFT在大图上的结果是否符合预期
int test_shiftFFt(std::string& err) {
	std::string img_path = R"(D:\DevelopMent\LibLSR20_Optimized\doc\CutOffFilter\srcImgs\20260902_090032\gray_Gaussian_Long WavelengthDUIBIAO Wrinkle 1A_SJ DUIBIAO wrinkle 1A_01_1_SZcO_IPP.png)";
	//img_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\srcImgs\3.png)";
	cv::Mat src = cv::imread(img_path, cv::IMREAD_UNCHANGED);
	businlog_error_return_err_reason(!src.empty(), "fail to imread image from path:" << img_path, err, -1);
	cv::Mat src32F;
	if (src.depth() == CV_64F) {
		src32F = src;
	}
	else {
		int type = CV_MAKE_TYPE(CV_64F, src.channels());
		src.convertTo(src32F, type);
	}
	int type = CV_MAKE_TYPE(CV_64F, src.channels());
	cv::Mat mask;
	std::string testContent; //测试项内容
	int choise = 2;
	switch (choise) {
	case 1:
		//此时所得dst与src完全相同，即：图片大小为32768*8063时，函数apply_frequency_filter的结果符合预期。
		mask = cv::Mat::ones(src.size(), type);
		testContent = "shitFFT";
		break;
	case 2:
	{
		//TODO::图片大小为32768*8063时，函数apply_frequency_filter的结果不符合预期。难度是因为图像尺寸过大，导致mask都接近0
		double sigma = 71; //数值越大，所得模糊程度越小。
		mask = gaussian_lowpass(src.rows, src.cols, sigma);
		testContent = "gaussian_lowpass";
	}
	break;
	case 3:
	{
	}
	break;
	}

	cv::Mat dst = apply_frequency_filter(src32F, mask);
	if (dst.depth() != src.depth()) {
		int type = CV_MAKE_TYPE(src.depth(), dst.channels());
		dst.convertTo(dst, type);
	}
	std::filesystem::path path_src(img_path);
	std::string out_path = path_src.parent_path().string() + "/" + path_src.stem().string() + "_" + testContent + ".png";
	businlog_error_return_err_reason(cv::imwrite(out_path, dst), "fail to write image to path:" << out_path, err, -1);
	businlog_info("success to write image to path:" << out_path);
	return 0;
}