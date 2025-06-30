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

int test_calcCovarMatrix(std::string& str_err_reason);

int test_skeleton(std::string& str_err_reason);

int test_steger(std::string& str_err_reason);

int test_mean_and_std_of_rows(std::string& str_err_reason);

//对小数点后的n位进行四舍五入
double round(double value, int n);
int test_HuaGeZi_and_calc_median(std::string& str_err_reason);


int test_getOnlyImgData();

int test_resize_images(std::string& str_err_reason);

//生成编译脚本
int test_generate_compilation_script(std::string& str_err_reason);
