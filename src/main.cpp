#include <iostream>
#include "test.h"
#include <json/json.h>
#include <iostream>
#include <filesystem>
#include "test_cpp.h"
#include <opencv2/core/cvdef.h>
int main(int argc, char* argv[])
{
	float angle = -90;
	float a = std::cos(angle*CV_PI/180);
	if (!std::filesystem::exists(RESULT_IMAGES_DIR))
	{
		std::filesystem::create_directories(RESULT_IMAGES_DIR);
	}
	while (true)
	{
		std::cout << "0、退出程序" << std::endl;
		std::cout << "1、演示灰度直方图calcHist" << std::endl;
		std::cout << "2、测试OpenCV::Parallel_for" << std::endl;
		std::cout << "3、统计图片在多种百分位情况下的标准差" << std::endl;
		std::cout << "4、将原图片中的两行合并为一行" << std::endl;
		std::cout << "5、测试自定义Allocate来初始化std::vector" << std::endl;
		std::cout << "6、测试minAreaRect" << std::endl;
		std::cout << "7、测试凸包：查找区域的凸点" << std::endl;
		std::cout << "8、测试霍夫直线变换" << std::endl;
		std::cout << "9、寻找两幅中差异点的坐标" << std::endl;
		std::cout << "10、计算连通区域的矩" << std::endl;
		std::cout << "11、elliptic_axis:计算连通区域的长短轴以及角度" << std::endl;
		std::cout << "12、计算协方差矩阵" << std::endl;
		std::cout << "13、计算区域骨架" << std::endl;
		std::cout << "14、Steger算法:提取二值图中曲线中心线" << std::endl;
		std::cout << "15、统计一幅图的行向均值的均值以及标准差" << std::endl;
		std::cout << "请输入您的选择:"; 
		int nChoise = -1;
		std::cin >> nChoise;
		int ret = -1;
		std::string str_err_reason;
		switch (nChoise)
		{
		case 0:
			std::cout << "成功退出程序" << std::endl;
			return 0;
		case 1:
			ret = test_demHist(str_err_reason);
			break;
		case 2:
			ret = test_parallel_for();
			break;
		case 3:
			ret = test_imgs_statistical_informations(str_err_reason);
			break;
		case 4:
			ret = test_towRows2oneRow(str_err_reason);
			break;
		case 5:
			ret = test_CustomAllocator(str_err_reason);
			break;
		case 6:
			ret = test_minAreaRect(str_err_reason);
			break;
		case 7:
			ret = test_convexhull(str_err_reason);
			break;
		case 8:
			ret = test_HoughLines(str_err_reason);
			break;
		case 9:
			ret = test_find_different_pnts(str_err_reason);
			break;
		case 10:
			ret = test_moment(str_err_reason);
			break;
		case 11:
			ret = test_elliptic_axis(str_err_reason);
			break;
		case 12:
			ret = test_calcCovarMatrix(str_err_reason);
			break;
		case 13:
			ret = test_skeleton(str_err_reason);
			break;
		case 14:
			ret = test_steger(str_err_reason);
			break;
		case 15:
			ret = test_mean_and_std_of_rows(str_err_reason);
			break;
		default:
			std::cout << "非法输入" << std::endl;
			continue;
		}
		if (ret)
		{
			std::cout << __FUNCTION__ << " | Has error ret:" << ret << ", reason:" << str_err_reason << std::endl;
		}
	}
	return 0;
}