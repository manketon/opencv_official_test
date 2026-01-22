#pragma once
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
template <class T>
void print_matrix(const std::vector<std::vector<T>>& vec)
{
	std::cout << "[";
	for (int i = 0; i < vec.size(); ++i)
	{
		const auto& vec_Row = vec[i];
		for (int j = 0; j < vec_Row.size(); ++j)
		{
			std::cout << vec_Row[j] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << "]" << std::endl;
}

inline void print_matrix(const cv::Mat& m)
{
	std::cout << m << std::endl;
}
template<class T>
void print_matrix(const std::vector<T>& vec)
{
	std::cout << "[";
	for (const auto& element : vec)
	{
		std::cout << element << ", ";
	}
	std::cout << "]" << std::endl;
}

template<typename T>
inline cv::Point2d area_center(const std::vector<T>& pnts)
{
	CV_Assert(pnts.empty() == false);
	double sumX = 0.0, sumY = 0.0;
	for (size_t i = 0; i < pnts.size(); ++i)
	{
		sumX += pnts[i].x;
		sumY += pnts[i].y;
	}
	cv::Point2d centroid;
	centroid.x = sumX / pnts.size();
	centroid.y = sumY / pnts.size();
	return centroid;
}