#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"
// =============================== 计算协方差矩阵 ============================
// 按行存储，以行为向量,输入矩阵mat为m行n列，则协方差矩阵covar为n行n列对称矩阵，均值mean为1行n列
template<typename _Tp>
int my_calcCovarMatrix(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& covar, std::vector<_Tp>& mean, bool scale = false)
{
	const int rows = mat.size();
	const int cols = mat[0].size();
	const int nsamples = rows;
	double scale_ = 1.;
	if (scale) scale_ = 1. / (nsamples /*- 1*/);

	covar.resize(cols);
	for (int i = 0; i < cols; ++i)
		covar[i].resize(cols, (_Tp)0);
	mean.resize(cols, (_Tp)0);

	for (int w = 0; w < cols; ++w) {
		for (int h = 0; h < rows; ++h) {
			mean[w] += mat[h][w];
		}
	}

	for (auto& value : mean) {
		value = 1. / rows * value;
	}

	for (int i = 0; i < cols; ++i) {
		std::vector<_Tp> col_buf(rows, (_Tp)0);
		for (int k = 0; k < rows; ++k)
			col_buf[k] = mat[k][i] - mean[i];

		for (int j = 0; j < cols; ++j) {
			double s0 = 0;
			for (int k = 0; k < rows; ++k) {
				s0 += col_buf[k] * (mat[k][j] - mean[j]);
			}
			covar[i][j] = (_Tp)(s0 * scale_);
		}
	}

	return 0;
}

int test_calcCovarMatrix(std::string& str_err_reason)
{
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	fprintf(stderr, "source matrix:\n");
	print_matrix(vec);

	fprintf(stderr, "\nc++ implement calculate covariance matrix:\n");
	std::vector<std::vector<float>> covar1;
	std::vector<float> mean1;
	if (my_calcCovarMatrix(vec, covar1, mean1, false/*true*/) != 0) {
		fprintf(stderr, "C++ implement calcCovarMatrix fail\n");
		return -1;
	}

	fprintf(stderr, "print covariance matrix: \n");
	print_matrix(covar1);
	fprintf(stderr, "print mean: \n");
	print_matrix(mean1);

	fprintf(stderr, "\nc++ opencv calculate covariance matrix:\n");
	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}
	//std::cout << mat << std::endl;

	//std::cout << "mat:" << std::endl << mat << std::endl;
	//cv::Mat means(1, mat.cols, mat.type(), cv::Scalar::all(0));
	//for (int i = 0; i < mat.cols; i++)
	//	means.col(i) = (cv::sum(mat.col(i)) / mat.rows).val[0];
	//std::cout << "means:" << std::endl << means << std::endl;
	//cv::Mat tmp = cv::repeat(means, mat.rows, 1);
	//cv::Mat mat2 = mat - tmp;
	//cv::Mat covar = (mat2.t()*mat2) / (mat2.rows /*- 1*/); // （X'*X)/n-1
	//std::cout << "covar:" << std::endl << covar << std::endl;

	cv::Mat covar2, mean2;
	cv::calcCovarMatrix(mat, covar2, mean2, cv::COVAR_NORMAL | cv::COVAR_ROWS/* | CV_COVAR_SCALE*/, CV_32FC1);
	fprintf(stderr, "print covariance matrix: \n");
	print_matrix(covar2);
	fprintf(stderr, "print mean: \n");
	print_matrix(mean2);

	return 0;
}
