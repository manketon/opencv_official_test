#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc.hpp>  
#include <opencv2/highgui.hpp>  
#include <opencv2/ximgproc.hpp>
#include <filesystem>
#include <iostream>
using namespace cv;

void skeletonize(const Mat& srcBin, Mat& dst) {
	// 先将图像转为二值图像  
	Mat binary = srcBin.clone();
	// 创建用作形态学操作的结构元素  
	Mat skel = Mat::zeros(binary.size(), CV_8UC1);
	Mat temp;
	Mat eroded;

	// 反复膨胀和腐蚀  
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	do {
		// 腐蚀操作  
		erode(binary, eroded, element);
		// 膨胀操作  
		dilate(eroded, temp, element);
		// 将骨架部分提取到skel中  
		subtract(binary, temp, temp);
		skel |= temp;
		// 更新binary为经过腐蚀后的图像  
		eroded.copyTo(binary);
	} while (countNonZero(binary) > 0);

	dst = skel;
}

// 定义条件，用于判断是否要删除像素  
bool skelCond(const Mat& img, int row, int col) {
	int p[8]; // 8个邻域像素  
	int z = 0; // 计数  
	int n = 0; // 连接数  

	// 邻域像素的索引  
	p[0] = img.at<uchar>(row - 1, col);     // 上  
	p[1] = img.at<uchar>(row - 1, col + 1); // 右上  
	p[2] = img.at<uchar>(row, col + 1);     // 右  
	p[3] = img.at<uchar>(row + 1, col + 1); // 右下  
	p[4] = img.at<uchar>(row + 1, col);     // 下  
	p[5] = img.at<uchar>(row + 1, col - 1); // 左下  
	p[6] = img.at<uchar>(row, col - 1);     // 左  
	p[7] = img.at<uchar>(row - 1, col - 1); // 左上  

	// 计算 z（非零邻居总数）  
	for (int i = 0; i < 8; ++i) {
		z += p[i] > 0 ? 1 : 0;
	}

	// 计算 n（连接数）  
	for (int i = 0; i < 8; ++i) {
		if (p[i] > 0 && p[(i + 1) % 8] == 0) {
			n++;
		}
	}

	// 骨架化的条件  
	return (z >= 2 && z <= 6 && n == 1 && img.at<uchar>(row, col) == 1);
}

void skeletonize2(const Mat& srcBin, Mat& dst) {
	Mat img = srcBin.clone();
	dst = Mat::zeros(srcBin.size(), CV_8UC1);

	// 迭代地删除像素直到无可删除像素  
	bool continues;
	do {
		continues = false;

		for (int i = 1; i < img.rows - 1; ++i) {
			for (int j = 1; j < img.cols - 1; ++j) {
				if (img.at<uchar>(i, j) == 1 && skelCond(img, i, j)) {
					dst.at<uchar>(i, j) = 1;
					img.at<uchar>(i, j) = 0; // 删除该像素  
					continues = true;
				}
			}
		}

		// 更新原图像，将dilated部分减去  
		bitwise_not(dst, img);
	} while (continues);

	// 将结果置为255用于显示  
	dst *= 255;
}


// Zhang-Suen 骨架化算法  
void zhangSuenSkeletonization(const Mat& src, Mat& dst) {
	// 创建输出图像，初始化为0  
	dst = Mat::zeros(src.size(), CV_8UC1);
	Mat temp;
	src.copyTo(temp);

	bool changes;

	do {
		changes = false;
		for (int r = 1; r < temp.rows - 1; r++) {
			for (int c = 1; c < temp.cols - 1; c++) {
				// 检测前景像素（即白色像素）  
				if (temp.at<uchar>(r, c) == 255) {
					// 计算邻域  
					int neighbors[8] = {
						temp.at<uchar>(r - 1, c - 1), temp.at<uchar>(r - 1, c),
						temp.at<uchar>(r - 1, c + 1), temp.at<uchar>(r, c + 1),
						temp.at<uchar>(r + 1, c + 1), temp.at<uchar>(r + 1, c),
						temp.at<uchar>(r + 1, c - 1), temp.at<uchar>(r, c - 1) };

					int p2 = neighbors[0] / 255;  // p2  
					int p3 = neighbors[1] / 255;  // p3  
					int p4 = neighbors[2] / 255;  // p4  
					int p5 = neighbors[3] / 255;  // p5  
					int p6 = neighbors[4] / 255;  // p6  
					int p7 = neighbors[5] / 255;  // p7  
					int p8 = neighbors[6] / 255;  // p8  
					int p9 = neighbors[7] / 255;  // p9  

					int A = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 255;  // 8-néighborhood  

					// Step 1  
					if (A >= 2 && A <= 6 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0) {
						temp.at<uchar>(r, c) = 0;  // 删除  
						changes = true;
					}
				}
			}
		}

		// 更新后处理以确保骨架中连通性  
		for (int r = 1; r < temp.rows - 1; r++) {
			for (int c = 1; c < temp.cols - 1; c++) {
				// 重置为当前图像  
				if (temp.at<uchar>(r, c) == 255) {
					// 检查并提取骨架  
					dst.at<uchar>(r, c) = 255;
				}
			}
		}

		// 继续步骤2  
		changes = false;
		temp.copyTo(dst);

		for (int r = 1; r < temp.rows - 1; r++) {
			for (int c = 1; c < temp.cols - 1; c++) {
				if (temp.at<uchar>(r, c) == 255) {
					// 获取邻域并进行统计  
					int neighbors[8] = {
						temp.at<uchar>(r - 1, c - 1), temp.at<uchar>(r - 1, c),
						temp.at<uchar>(r - 1, c + 1), temp.at<uchar>(r, c + 1),
						temp.at<uchar>(r + 1, c + 1), temp.at<uchar>(r + 1, c),
						temp.at<uchar>(r + 1, c - 1), temp.at<uchar>(r, c - 1) };

					int p2 = neighbors[0] / 255;
					int p3 = neighbors[1] / 255;
					int p4 = neighbors[2] / 255;
					int p5 = neighbors[3] / 255;
					int p6 = neighbors[4] / 255;
					int p7 = neighbors[5] / 255;
					int p8 = neighbors[6] / 255;
					int p9 = neighbors[7] / 255;

					int A = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) / 255;

					// Step 2  
					if (A >= 2 && A <= 6 && p2 * p6 * p8 == 0 && p2 * p4 * p8 == 0) {
						temp.at<uchar>(r, c) = 0; // 删除  
						changes = true;
					}
				}
			}
		}

	} while (changes); // 当仍有变化时继续  

	// 骨架图像变换为 uint8 类型显示  
	dst *= 255;
}

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 */
void thinningIteration(Mat& im, int iter)
{
	Mat marker = Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}

/**
 * Function for thinning the given binary image
 */
void thinning(Mat& im)
{
	im /= 255;

	Mat prev = Mat::zeros(im.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (countNonZero(diff) > 0);

	im *= 255;
}

//与Halcon的结果还有一些差距
void skeletonization4(const cv::Mat srcBin, cv::Mat& dst)
{
// 	if (inputImage.empty())
// 		std::cout << "Inside skeletonization, Source empty" << std::endl;
// 
// 	Mat outputImage;
// 	cvtColor(inputImage, outputImage, CV_BGR2GRAY);
// 	threshold(outputImage, outputImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
	dst = srcBin.clone();
	thinning(dst);
}



void test_1(const cv::Mat& srcBin, cv::Mat& dst)
{
	skeletonize(srcBin, dst);
}
void test_2(const cv::Mat& srcBin, cv::Mat& dst)
{
	auto tempBin = srcBin / 255;
	skeletonize2(tempBin, dst);
}
void test_3(const cv::Mat& srcBin, cv::Mat& dst)
{
	zhangSuenSkeletonization(srcBin, dst);
}
void test_4(const cv::Mat& srcBin, cv::Mat& dst)
{//与cv::ximgproc::THINNING_ZHANGSUEN一致。
	skeletonization4(srcBin, dst);
}

void test_5(const cv::Mat& srcBin, cv::Mat& dst)
{
	//cv::ximgproc::THINNING_ZHANGSUEN过于冗余
	cv::ximgproc::thinning(srcBin, dst, cv::ximgproc::THINNING_GUOHALL);
}

void ThinSubiteration1(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						/// calculate criteria 3
						int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
						if (c3 == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}


void ThinSubiteration2(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
						if (E == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}


void normalizeLetter(Mat& inputarray, Mat& outputarray) {
	bool bDone = false;
	int rows = inputarray.rows;
	int cols = inputarray.cols;

	inputarray.convertTo(inputarray, CV_32FC1);

	inputarray.copyTo(outputarray);

	outputarray.convertTo(outputarray, CV_32FC1);

	/// pad source
	Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
	for (int i = 0; i < (rows + 2); i++) {
		p_enlarged_src.at<float>(i, 0) = 0.0f;
		p_enlarged_src.at<float>(i, cols + 1) = 0.0f;
	}
	for (int j = 0; j < (cols + 2); j++) {
		p_enlarged_src.at<float>(0, j) = 0.0f;
		p_enlarged_src.at<float>(rows + 1, j) = 0.0f;
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (inputarray.at<float>(i, j) >= 20.0f) {
				p_enlarged_src.at<float>(i + 1, j + 1) = 1.0f;
			}
			else
				p_enlarged_src.at<float>(i + 1, j + 1) = 0.0f;
		}
	}

	/// start to thin
	Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

	while (bDone != true) {
		/// sub-iteration 1
		ThinSubiteration1(p_enlarged_src, p_thinMat1);
		/// sub-iteration 2
		ThinSubiteration2(p_thinMat1, p_thinMat2);
		/// compare
		compare(p_enlarged_src, p_thinMat2, p_cmp, cv::CMP_EQ);
		/// check
		int num_non_zero = countNonZero(p_cmp);
		if (num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		/// copy
		p_thinMat2.copyTo(p_enlarged_src);
	}
	// copy result
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outputarray.at<float>(i, j) = p_enlarged_src.at<float>(i + 1, j + 1);
		}
	}
}
void test_6(const cv::Mat& srcBin, cv::Mat& dst)
{
	cv::Mat tmpBin = srcBin.clone();
	normalizeLetter(tmpBin, dst);
}

int test_skeleton(std::string& str_err_reason) 
{
	std::string str_src_bin_path = "D:/DevelopMent/LibLSR20_Optimized/testImg/only_one_region/one_region.png";
	std::cout << "请输入二值图路径:";
	std::cin >> str_src_bin_path;
	cv::Mat srcBin = cv::imread(str_src_bin_path, cv::IMREAD_UNCHANGED);
	CV_Assert(srcBin.empty() == false && srcBin.type() == CV_8UC1);
	
	while (true)
	{
		cv::TickMeter tm;
		Mat dst;
		int Algo = 0;
		std::cout << "请输入所用算法(1至6，0为退出):";
		std::cin >> Algo;
		tm.start();
		switch (Algo)
		{
		case 0:
			std::cout << __FUNCTION__ << " | 成功退出" << std::endl;
			return 0;
		case 1:
			test_1(srcBin, dst);
			break;
		case 2:
			test_2(srcBin, dst);
			break;
		case 3:
			test_3(srcBin, dst);
			break;
		case 4:
			test_4(srcBin, dst);
			break;
		case 5:
			test_5(srcBin, dst);
			break;
		case 6:
			test_6(srcBin, dst);
			break;
		default:
			break;
		}
		tm.stop();
		std::cout << __FUNCTION__ << " | cost " << tm.getTimeSec() << "S" << std::endl;
		std::filesystem::path path_src(str_src_bin_path);
		std::string str_dst_path = path_src.parent_path().string() + "/" + path_src.stem().string() + "_algo"+ std::to_string(Algo) + "_skeleton.png";
		std::vector<cv::Mat> channels{ srcBin, dst, srcBin };
		cv::Mat rslt;
		cv::merge(channels, rslt);
		CV_Assert(cv::imwrite(str_dst_path, rslt));
	}
	return 0;
}