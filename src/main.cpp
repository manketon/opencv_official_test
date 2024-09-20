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
		std::cout << "0���˳�����" << std::endl;
		std::cout << "1����ʾ�Ҷ�ֱ��ͼcalcHist" << std::endl;
		std::cout << "2������OpenCV::Parallel_for" << std::endl;
		std::cout << "3��ͳ��ͼƬ�ڶ��ְٷ�λ����µı�׼��" << std::endl;
		std::cout << "4����ԭͼƬ�е����кϲ�Ϊһ��" << std::endl;
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
			break;
		case 6:
			break;
		case 7:
			break;
		case 8:
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
