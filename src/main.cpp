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
		std::cout << "0���˳�����" << std::endl;
		std::cout << "1����ʾ�Ҷ�ֱ��ͼcalcHist" << std::endl;
		std::cout << "2������OpenCV::Parallel_for" << std::endl;
		std::cout << "3��ͳ��ͼƬ�ڶ��ְٷ�λ����µı�׼��" << std::endl;
		std::cout << "4����ԭͼƬ�е����кϲ�Ϊһ��" << std::endl;
		std::cout << "5�������Զ���Allocate����ʼ��std::vector" << std::endl;
		std::cout << "6������minAreaRect" << std::endl;
		std::cout << "7������͹�������������͹��" << std::endl;
		std::cout << "8�����Ի���ֱ�߱任" << std::endl;
		std::cout << "9��Ѱ�������в���������" << std::endl;
		std::cout << "10��������ͨ����ľ�" << std::endl;
		std::cout << "11��elliptic_axis:������ͨ����ĳ������Լ��Ƕ�" << std::endl;
		std::cout << "12������Э�������" << std::endl;
		std::cout << "13����������Ǽ�" << std::endl;
		std::cout << "14��Steger�㷨:��ȡ��ֵͼ������������" << std::endl;
		std::cout << "15��ͳ��һ��ͼ�������ֵ�ľ�ֵ�Լ���׼��" << std::endl;
		std::cout << "����������ѡ��:"; 
		int nChoise = -1;
		std::cin >> nChoise;
		int ret = -1;
		std::string str_err_reason;
		switch (nChoise)
		{
		case 0:
			std::cout << "�ɹ��˳�����" << std::endl;
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
			std::cout << "�Ƿ�����" << std::endl;
			continue;
		}
		if (ret)
		{
			std::cout << __FUNCTION__ << " | Has error ret:" << ret << ", reason:" << str_err_reason << std::endl;
		}
	}
	return 0;
}