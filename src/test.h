/*****************************************************************//**
 * @file   test.h
 * @brief
 *
 * @author tunicorn
 * @date   October 2022
 *********************************************************************/
#pragma once
#include <string>
#ifndef DEBUG_IMGS_DIR
#define RESULT_IMAGES_DIR ("./result_imgs/")
#endif // DEBUG_IMGS_DIR
/**
 * @brief  
 */
int test_decompose_homography(int argc, char* argv[]);

int test_Sobel(int argc, char** argv);

int test_load_img_from_bytes();

//演示直方图
int test_demHist(std::string& str_err_reason);

int test_parallel_for();

int test_imgs_statistical_informations(std::string& str_err_reason);

int test_towRows2oneRow(std::string& str_err_reason);

int test_minAreaRect(std::string& str_err_reason);

int test_convexhull(std::string& str_err_reason);

int test_HoughLines(std::string& str_err_reason);
//寻找两幅图中不一样的像素点坐标
int test_find_different_pnts(std::string& str_err_reason);

int test_moment(std::string& str_err_reason);
int test_elliptic_axis(std::string& str_err_reason);
