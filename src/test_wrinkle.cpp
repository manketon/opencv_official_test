#include "cutOffFilterImpl.hpp"
#include <string>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
using namespace nsLSR20::nsCutOffFilter;
int test_wrinkle()
{
	try
	{
		// 图像尺寸
		const int W = 32768;
		const int H = 8063;
		const int OuterRadius = 48250; //晶圆最外围尺寸半径
		const int InnerRadius = 16002; //晶圆最内围尺寸半径
		const int StepSize = 4;
		std::string in_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\srcImgs\DUIBIAO Wrinkle 1A_SJ DUIBIAO wrinkle 1A_01_1_SZcO_IPP.png)";
		//string out_path = "result_band.png";
		std::string out_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\rslt\result_DUIBIAO Wrinkle 1A_SJ DUIBIAO wrinkle 1A_01_1_SZcO_IPP.png)";
		//string out_path = "result_Uniform_band.png";

		cv::Mat srcImg = cv::imread(in_path, cv::IMREAD_UNCHANGED);
		if (srcImg.empty())
		{
			std::cerr << "错误：读取图像失败！" << std::endl;
			return -1;
		}
		// ========== 切换此处参数即可更换滤波 ==========
		// 示例1：高斯低通
		//FilterType fType = FILTER_GAUSS;
		//FilterMode fMode = MODE_LOW;

		//// 示例2：高斯高通
		FilterType fType = FILTER_GAUSS;
		FilterMode fMode = MODE_HIGH;

		// 示例3：Uniform低通
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_LOW;

		//// 示例4：Uniform高通
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_HIGH;

		//// 示例5：Uniform带通（波长50~150）
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_BAND;

		// 波长参数（可自行修改）
		double bandLowLam = 100.0; //如果没有选中Uniform带通模式，则不使用这两个参数。默认参数设为1。
		double bandHighLam = 250.0; //如果没有选中Uniform带通模式，则不使用这两个参数。默认参数设为1000。

		// 构造对象，传入全部动态参数（内部自动校验）
		//RowFFTMultiFilter filter(W, H, targetWave, bandLowLam, bandHighLam);
		RowFFTMultiFilter filter(W, H, bandLowLam, bandHighLam);
		cv::Mat dstImg;
		cv::TickMeter tm;
		tm.start();
		int ret = filter.runProcess(srcImg, OuterRadius, InnerRadius, StepSize, fType, fMode, dstImg);
		tm.stop();
		std::cout << "处理时间：" << tm.getTimeMilli() << "ms" << std::endl;
		if (ret != 0)
		{
			std::cerr << "图像处理失败！" << std::endl;
			return -1;
		}
		auto bSuccess = cv::imwrite(out_path, dstImg);
		std::cout << (bSuccess ? "成功" : "失败") << "地保存文件到路径:" << out_path << std::endl;
	}
	catch (const std::invalid_argument& e)
	{
		std::cerr << "参数校验失败：" << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cerr << "程序发生未知异常！" << std::endl;
		return -1;
	}
	return 0;
}