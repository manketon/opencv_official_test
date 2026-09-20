/******************************************************************
* Copyright (c) 2022-2099, GongSiTest Inc.
* All rights reserved.
* @file: utils.hpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:    minglu
* @version: 1.0
* @date: 2026/09/20
*
* @see
*
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本    <th>日期        <th>作者    <th>备注 </tr>
*  <tr> <td>1.0     <td>2026/09/20  <td>minglu   <td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once
#include <opencv2/core/core.hpp>
#include <busin_log.h>
#include <opencv_tool/opencv_common_tool.h>

#ifdef __cplusplus
extern "C" {
	//包含C语言接口、定义或头文件
#endif
#ifdef __cplusplus
}
#endif
//引用C++头文件：先是标准库头文件，后是项目头文件

//宏定义

//类型定义

//函数原型定义
namespace nsTest {
	inline bool makeInMatNotChange(const cv::Size& imgSize, cv::Rect& roi)
	{
		if (imgSize.width < roi.width || imgSize.height < roi.height) {//不能够放下一个ROI
			return false;
		}
		if (roi.x < 0) {
			roi.x = 0;
		}
		else if (roi.x + roi.width > imgSize.width) {
			roi.x = imgSize.width - roi.width;
		}
		if (roi.y < 0) {
			roi.y = 0;
		}
		else if (roi.y + roi.height > imgSize.height) {
			roi.y = imgSize.height - roi.height;
		}
		return true;
	}
	struct ImgSliceRslt
	{
		cv::Mat subImg; //子图
		cv::Rect roi; //子图在原图中的位置
	};

	static int sliceImage(const cv::Mat& srcImg, const int subImgWidth, const int subImgHeight
		, const int XOverlappedLen, const int YOverlappedLen, std::vector<ImgSliceRslt>& slicedRslts, std::string& err)
	{
		businlog_error_return_is_greaterequal(XOverlappedLen, 0, "", err, -1);
		businlog_error_return_is_greaterequal(YOverlappedLen, 0, "", err, -2);
		businlog_error_return_is_greater(subImgHeight, YOverlappedLen, "", err, -3);
		businlog_error_return_is_greater(subImgWidth, XOverlappedLen, "", err, -4);
		businlog_error_return_is_lessequal(subImgHeight, srcImg.rows, "", err, -5);
		businlog_error_return_is_lessequal(subImgWidth, srcImg.cols, "", err, -6);
		slicedRslts.clear();
		const cv::Size wholeImgSize = srcImg.size();
		slicedRslts.reserve((srcImg.cols / (subImgWidth - XOverlappedLen) + 1) * (srcImg.rows / (subImgHeight - YOverlappedLen) + 1));
		for (int i = 0; i < srcImg.rows; i += subImgHeight - YOverlappedLen)
		{
			bool bExistRowOutImg = false;
			if (i + subImgHeight > srcImg.rows)
			{
				i = srcImg.rows - subImgHeight;
				bExistRowOutImg = true;
			}
			for (int j = 0; j < srcImg.cols; j += subImgWidth - XOverlappedLen)
			{
				cv::Rect roi(j, i, subImgWidth, subImgHeight);
				bool bExistColOutImg = false;  //是否存在ROI越出图片边界的情况

				if (!sp_cv::is_in_mat(srcImg, roi))
				{
					bExistColOutImg = true;
					businlog_error_return_err_reason(makeInMatNotChange(wholeImgSize, roi), "fail to make ROI in Mat, wholeImgSize:" << wholeImgSize << ", roi:" << roi
						, err, -7);
				}
				cv::Mat subImg(srcImg, roi);
				ImgSliceRslt rslt;
				rslt.subImg = subImg, rslt.roi = roi;
				slicedRslts.emplace_back(rslt);
				if (bExistColOutImg)
				{//遇到了边界
					break;
				}
				if (j + subImgWidth == srcImg.cols)
				{//刚好够一个完整的ROI
					break;
				}
			}//end for
			if (bExistRowOutImg)
			{
				break;
			}
		}//end for
		return 0;
	}
}//nsTest