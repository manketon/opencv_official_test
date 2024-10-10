/******************************************************************
* Copyright (c) 2022-2099, AVANT Inc.
* All rights reserved.
* @file: test_cpp.h
* @brief: 用于测试C++语法
* @author:    minglu
* @version: 1.0
* @date: 2024/10/10
* 
* @see
* 
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本    <th>日期        <th>作者    <th>备注 </tr>
*  <tr> <td>1.0     <td>2024/10/10  <td>minglu   <td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once
#include <string>

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
int test_CustomAllocator(std::string& str_err_reason);
