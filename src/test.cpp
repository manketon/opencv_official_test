#include <opencv2/opencv.hpp>
int testParallel_for()
{
	return 0;
}

int test_load_img_from_bytes()
{
	std::string str_src_img_path = "D:/wafer_images/Si PSL 83nm_SSc/Si PSL 83nm_SSc";
	int nImg_width = 32768;
	int nImg_height = 6001;
	//! 以二进制流方式读取图片到内存
	FILE* pFile = fopen(str_src_img_path.c_str(), "rb");
	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);
	CV_Assert(lSize % 2 == 0);
	rewind(pFile);
	const long pixel_num = lSize / 2;
	CV_Assert(pixel_num == nImg_height * nImg_width);
	ushort* pData = new ushort[pixel_num];
	fread(pData, sizeof(ushort), pixel_num, pFile);
	fclose(pFile);
	cv::Mat img(nImg_height, nImg_width, CV_16UC1);
	for (int i = 0; i < nImg_height; ++i)
	{
		for (int j = 0; j < nImg_width; ++j)
		{
			img.at<ushort>(i, j) = pData[i * nImg_width + j];
		}
	}
	delete[] pData;
	cv::imwrite("D:/wafer_images/Si PSL 83nm_SSc/rslt.png", img);
	return 0;
}
