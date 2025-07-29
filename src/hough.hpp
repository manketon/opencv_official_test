/******************************************************************
* Copyright (c) 2022-2099, AVANT Inc.
* All rights reserved.
* @file: hough.hpp
* @brief: ���˵���ļ����ܡ���; (Comment)��
* @author:    minglu
* @version: 1.0
* @date: 2025/07/29
* 
* @see
* 
* <b>�汾��¼��</b><br>
* <table>
*  <tr> <th>�汾    <th>����        <th>����    <th>��ע </tr>
*  <tr> <td>1.0     <td>2025/07/29  <td>minglu   <td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once

#ifdef __cplusplus  
extern "C" {
	//����C���Խӿڡ������ͷ�ļ�
#endif
#ifdef __cplusplus  
}
#endif  
//����C++ͷ�ļ������Ǳ�׼��ͷ�ļ���������Ŀͷ�ļ�
#include <opencv2/core.hpp>
//�궨��

//���Ͷ���
namespace YRP
{
	void HoughLinesP(cv::InputArray _image, cv::OutputArray _lines,
		double rho, double theta, int threshold,
		double minLineLength, double maxGap);
}
//����ԭ�Ͷ���
