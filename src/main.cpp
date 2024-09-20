#include <iostream>
#include "test.h"
#include <json/json.h>
#include <iostream>
int main(int argc, char* argv[])
{
	int ret = -1;
	//Json::Value jv1, jv2;
	//std::cout << std::boolalpha << (jv1 == jv2) << std::endl;
	//ret = test_decompose_homography(argc, argv);
	//ret = test_Sobel(argc, argv);
	//ret = test_load_img_from_bytes();
	
	while (true)
	{
		std::cout << "0、退出程序" << std::endl;
		std::cout << "1、演示灰度直方图calcHist" << std::endl;
		std::cout << "2、测试OpenCV::Parallel_for" << std::endl;
		std::cout << "3、统计图片在多种百分位情况下的标准差" << std::endl;
		std::cout << "4、将原图片中的两行合并为一行" << std::endl;
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
			break;
		case 6:
			break;
		case 7:
			break;
		case 8:
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
