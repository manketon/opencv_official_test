/******************************************************************
* Copyright (c) 2022-2099, GongSiTest Inc.
* All rights reserved.
* @file: minAreaRectExt.h
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:    minglu
* @version: 1.0
* @date: 2026/05/21
*
* @see
*
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本    <th>日期        <th>作者    <th>备注 </tr>
*  <tr> <td>1.0     <td>2026/05/21  <td>minglu   <td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once
#include <opencv2/core/mat.hpp>

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
cv::RotatedRect minAreaRectExt(cv::InputArray _points);