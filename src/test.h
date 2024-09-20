/*****************************************************************//**
 * @file   test.h
 * @brief
 *
 * @author tunicorn
 * @date   October 2022
 *********************************************************************/
#pragma once
/**
 * @brief  
 */
int test_decompose_homography(int argc, char* argv[]);

int test_Sobel(int argc, char** argv);

int test_load_img_from_bytes();

//ÑÝÊ¾Ö±·½Í¼
int test_demHist(std::string& str_err_reason);

int test_parallel_for();

int test_imgs_statistical_informations(std::string& str_err_reason);

int test_towRows2oneRow(std::string& str_err_reason);