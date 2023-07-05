#include <iostream>
#include "test.h"
#include <json/json.h>
#include <iostream>
int main(int argc, char* argv[])
{
	int ret = -1;
	Json::Value jv1, jv2;
	std::cout << (jv1 == jv2) << std::endl;
	//ret = test_decompose_homography(argc, argv);
	//ret = test_Sobel(argc, argv);
	ret = test_load_img_from_bytes();
	return 0;
}
