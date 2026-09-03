/******************************************************************
* Copyright (c) 2022-2099, GongSiTest Inc.
* All rights reserved.
* @file: cutOffFilterImpl.hpp
* @brief: 简短说明文件功能、用途 (Comment)。
* @author:    minglu
* @version: 1.0
* @date: 2026/08/11
*
* @see
*
* <b>版本记录：</b><br>
* <table>
*  <tr> <th>版本    <th>日期        <th>作者    <th>备注 </tr>
*  <tr> <td>1.0     <td>2026/08/11  <td>minglu   <td>Create head file </tr>
* </table>
*****************************************************************/
#pragma once

#ifdef __cplusplus
extern "C" {
	//包含C语言接口、定义或头文件
#endif
#ifdef __cplusplus
}
#endif
//引用C++头文件：先是标准库头文件，后是项目头文件
#include <fftw3.h>
#include <opencv2/core.hpp>
#include <string>
#include <iomanip>
#include <omp.h>
#ifdef OUTPUT_DEBUG_INFO
#include <iostream>
#endif
#include <busin_log.h>
//宏定义

//类型定义
namespace nsLSR20 {
	namespace nsCutOffFilter {
		// 滤波类型（即界面中的Window Function）
		enum FilterType
		{
			FILTER_GAUSS,  //高斯
			FILTER_UNIFORM,  //uniform
			FILTER_NONE //None
		};
		// 滤波模式（即界面中的Cut-Off Filter Type）
		enum FilterMode
		{
			MODE_LOW, //低通（即Short Wavelength）
			MODE_HIGH, //高通 （即Long Wavelength）
			MODE_BAND  //带通 （即Band Pass）
		};

		class RowFFTMultiFilter
		{
		private:
			// 动态输入参数（构造传入）
			const int N_POINT_PER_ROW;  // 每行采样点数
			const int M_ROW_COUNT;  // 总行数
			const double LAMBDA_BAND_LOW; //低通波长的值 （即Filter Cut-Off Lower Limits，滤波器类型为Long Wavelength时，则不用设置此值）
			const double LAMBDA_BAND_HIGH; //高通波长的值（即Filter Cut-Off Upper Limits，滤波器类型为Short Wavelength时，则不用设置此值）
			const int m_max_threads;
			// ========== 优化新增：单线程独立FFTW资源结构体 ==========
			struct ThreadFFTResource
			{
				double* real_buf;
				fftw_complex* cplx_buf;
				fftw_plan plan_r2c;
				fftw_plan plan_c2r;
			};
			// 存储所有线程独立资源，构造时一次性预分配
			std::vector<ThreadFFTResource> m_thread_resources;

			/**
			 * 1D 频谱中心化 fftshift：低频挪到数组中间
			 * @param cplx 复数频域数组
			 */
			void fftShift1D(fftw_complex* cplx)
			{
				int half = N_POINT_PER_ROW / 2;
				std::vector<fftw_complex> temp(half);
				// 1. 拷贝后半段到临时缓存
				for (int i = 0; i < half; i++)
				{
					temp[i][0] = cplx[half + i][0];
					temp[i][1] = cplx[half + i][1];
				}
				// 2. 前半段移至数组后半区
				for (int i = 0; i < half; i++)
				{
					cplx[half + i][0] = cplx[i][0];
					cplx[half + i][1] = cplx[i][1];
				}
				// 3. 临时后半段放到前半区，完成中心化
				for (int i = 0; i < half; i++)
				{
					cplx[i][0] = temp[i][0];
					cplx[i][1] = temp[i][1];
				}
			}

			/**
			 * 逆中心化 ifftshift，fftshift逆操作
			 */
			void ifftShift1D(fftw_complex* cplx)
			{
				fftShift1D(cplx);
			}

			/**
			 * 核心频域透过率计算，按规则区分波长参数使用逻辑
			 * 1. GAUSS LOW/HIGH：仅使用LAMBDA_BAND_LOW / LAMBDA_BAND_HIGH
			 * 2. UNIFORM LOW/HIGH：仅使用LAMBDA_BAND_LOW / LAMBDA_BAND_HIGH
			 * 3. UNIFORM BAND：仅使用LAMBDA_BAND_LOW / LAMBDA_BAND_HIGH
			 */
			double calcTransmittance(double f, FilterType fType, FilterMode fMode)
			{
				// 高斯滤波：
				if (fType == FILTER_GAUSS)
				{
					double sqrt_2ln2 = sqrt(2 * log(2.0));
					if (fMode == MODE_LOW || fMode == MODE_HIGH)
					{
						double f0 = 0.0;
						if (fMode == MODE_LOW)
							f0 = 1.0 / LAMBDA_BAND_LOW;
						if (fMode == MODE_HIGH)
							f0 = 1.0 / LAMBDA_BAND_HIGH;

						double SIGMA_F = f0 / sqrt_2ln2;
						double H_low = exp(-(f * f) / (2.0 * SIGMA_F * SIGMA_F));
						return (fMode == MODE_LOW) ? H_low : (1.0 - H_low);
					}
					else if (fMode == MODE_BAND)
					{
						double f0 = 1.0 / LAMBDA_BAND_LOW; //上限
						double f1 = 1.0 / LAMBDA_BAND_HIGH; // 下限

						double SIGMA_F0 = f0 / sqrt_2ln2;
						double SIGMA_F1 = f1 / sqrt_2ln2;
						double H_low0 = exp(-(f * f) / (2.0 * SIGMA_F0 * SIGMA_F0));
						double H_low1 = exp(-(f * f) / (2.0 * SIGMA_F1 * SIGMA_F1));
						double result_H = H_low0 - H_low1;
						return (result_H > 0) ? result_H : 0.0;
					}
				}
				else
				{	// Uniform 滤波
					double abs_f = fabs(f);
					// 低通：
					if (fMode == MODE_LOW)
					{
						double f0 = 1.0 / LAMBDA_BAND_LOW;
						double diff = abs_f - f0;
						return 1.0 / (1.0 + std::exp(1300 * diff)); //此处1300为经验值，可调整
					}
					// 高通：
					else if (fMode == MODE_HIGH)
					{
						double f0 = 1.0 / LAMBDA_BAND_HIGH;
						double diff = abs_f - f0;
						return 1.0 / (1.0 + std::exp(-1300 * diff)); //此处1300为经验值，可调整
					}
					// 带通：使用band区间波长
					else if (fMode == MODE_BAND)
					{
						double f_band_low = 1.0 / LAMBDA_BAND_HIGH;
						double f_band_high = 1.0 / LAMBDA_BAND_LOW;
						double diff_low = abs_f - f_band_low;
						double diff_high = abs_f - f_band_high;
						return 1.0 / ((1.0 + std::exp(1500 * diff_low)) * (1.0 + std::exp(-1500 * diff_high)));   //此处1500为经验值，可调整
					}
				}
				return 1.0;
			}

			/**
			 * 单行标准完整流程：
			 * 输入像素 → FFT → fftshift中心化 → 高斯频域滤波 → ifftshift → IFFT → 输出浮点
			 * @param row_in 单行原始图像数据
			 * @param dx_step 当前行扫描步距
			 * @param row_out 输出滤波后double数组
			 */

			void processSingleRow(int row_idx, const cv::Mat& img_src, double dx, FilterType filterType, FilterMode filterMode, std::vector<double>& row_out, ThreadFFTResource& res)
			{
				const ushort* row_ptr = img_src.ptr<ushort>(row_idx);
				// 1. 图像ushort转double，SIMD向量化
				for (int x = 0; x < N_POINT_PER_ROW; x++)
				{
					res.real_buf[x] = static_cast<double>(row_ptr[x]);
				}
				// 2. 正向FFT（使用线程私有plan）
				fftw_execute(res.plan_r2c);
				// 3. 频谱中心化，复用线程预分配临时缓存
				fftShift1D(res.cplx_buf);
				int half = N_POINT_PER_ROW / 2;
				for (int k = 0; k < N_POINT_PER_ROW; k++)
				{
					// 中心化后频率偏移：中心k=half对应f=0
					int delta_k = k - half;
					double f = delta_k * dx; // 真实物理频率(正负)
					double H = calcTransmittance(f, filterType, filterMode);
					// 频域复数乘以高斯透过率
					double re = res.cplx_buf[k][0];
					double im = res.cplx_buf[k][1];
					res.cplx_buf[k][0] = re * H;
					res.cplx_buf[k][1] = im * H;
				}
				// 5. 逆中心化
				ifftShift1D(res.cplx_buf);
				// 6. 逆FFT
				fftw_execute(res.plan_c2r);
				// 7. IFFT归一化，提取实部输出
				row_out.resize(N_POINT_PER_ROW);
				double scale = 1.0 / N_POINT_PER_ROW;
				for (int x = 0; x < N_POINT_PER_ROW; x++)
				{
					row_out[x] = res.real_buf[x] * scale;
				}
			}

			/**
			 * 浮点二维数组全局归一化映射到 uint16 [0,4095]
			 */
			void floatToUint16(const std::vector<std::vector<double>>& float_buf, cv::Mat& dst_u16)
			{
				dst_u16.create(M_ROW_COUNT, N_POINT_PER_ROW, CV_16UC1);
				double global_min = 1e15;
				double global_max = -1e15;

				// ===== 并行求全局 min max =====
#pragma omp parallel for num_threads(m_max_threads) schedule(static)
				for (int y = 0; y < M_ROW_COUNT; y++)
				{
					// 每个线程独立局部min/max
					double loc_min = 1e15;
					double loc_max = -1e15;
					const auto& row = float_buf[y];
					for (int x = 0; x < N_POINT_PER_ROW; x++)
					{
						double v = row[x];
						if (v < loc_min) loc_min = v;
						if (v > loc_max) loc_max = v;
					}
					// 临界区合并到全局min/max
#pragma omp critical
					{
						if (loc_min < global_min) global_min = loc_min;
						if (loc_max > global_max) global_max = loc_max;
					}
				}

				double range = fabs(global_max - global_min) < 1e-9 ? 1.0 : (global_max - global_min);
				const size_t MAX_VALUE = 4095/*65535*/;
				double scale = MAX_VALUE / range;
#pragma omp parallel for num_threads(m_max_threads) schedule(static)
				for (int y = 0; y < M_ROW_COUNT; y++)
				{
					const auto& row = float_buf[y];
					ushort* pDst_row = dst_u16.ptr<ushort>(y);
					for (int x = 0; x < N_POINT_PER_ROW; x++)
					{
						double norm = (row[x] - global_min) * scale;
						norm = norm < 0.0 ? 0.0 : (norm > MAX_VALUE ? MAX_VALUE : norm);
						pDst_row[x] = static_cast<ushort>(round(norm));
					}
				}
			}

		public:
			/**
			 * 构造函数：全部关键参数外部传入 + 参数合法性校验
			 * @param nPerRow 每行采样点数
			 * @param totalRow 图像总行数
			 * @param bandLow 带通下限波长
			 * @param bandHigh 带通上限波长
			 */
			RowFFTMultiFilter(int nPerRow, int totalRow, double bandLow, double bandHigh)
				: N_POINT_PER_ROW(nPerRow),
				M_ROW_COUNT(totalRow),
				LAMBDA_BAND_LOW(bandLow),
				LAMBDA_BAND_HIGH(bandHigh),
				m_max_threads(omp_get_max_threads())
			{
				// ========== 参数合法性校验 ==========
				if (N_POINT_PER_ROW <= 0)
					throw std::invalid_argument("每行采样点数 N_POINT_PER_ROW 必须大于0");
				if (M_ROW_COUNT <= 0)
					throw std::invalid_argument("总行数 M_ROW_COUNT 必须大于0");

				// ========== 构造阶段串行预分配所有线程FFTW资源 ==========
				m_thread_resources.resize(m_max_threads);
				for (int t = 0; t < m_max_threads; t++)
				{
					auto& res = m_thread_resources[t];
					// FFTW对齐内存分配
					res.real_buf = fftw_alloc_real(N_POINT_PER_ROW);
					if (nullptr == res.real_buf) {
						throw std::runtime_error("FFTW: fail to malloc for real_buf");
					}
					res.cplx_buf = fftw_alloc_complex(N_POINT_PER_ROW);
					if (nullptr == res.cplx_buf) {
						throw std::runtime_error("FFTW: fail to malloc for cplx_buf");
					}
					res.plan_r2c = fftw_plan_dft_r2c_1d(N_POINT_PER_ROW, res.real_buf, res.cplx_buf, FFTW_MEASURE);
					res.plan_c2r = fftw_plan_dft_c2r_1d(N_POINT_PER_ROW, res.cplx_buf, res.real_buf, FFTW_MEASURE);
				}
			}

			// 析构自动释放FFTW内存
			~RowFFTMultiFilter()
			{
				// 顺序释放每个线程的FFTW plan与内存
				for (auto& res : m_thread_resources)
				{
					fftw_destroy_plan(res.plan_r2c);
					fftw_destroy_plan(res.plan_c2r);
					if (res.real_buf) {
						fftw_free(res.real_buf);
						res.real_buf = nullptr;
					}
					if (res.cplx_buf) {
						fftw_free(res.cplx_buf);
						res.cplx_buf = nullptr;
					}
				}
			}

			/**
			 * 主处理接口
			 */
			int runProcess(const cv::Mat& srcImg,
				const int OuterRadius,
				const int InnerRadius,
				const int StepSize,
				const FilterType filterType,
				const FilterMode filterMode,
				cv::Mat& dstImg)
			{
				if (filterType == FilterType::FILTER_NONE) {
					dstImg = srcImg.clone();
					return 0;
				}
				if (LAMBDA_BAND_LOW >= LAMBDA_BAND_HIGH && filterMode == FilterMode::MODE_BAND)
				{
					//cerr << "错误：带通波长必须满足 LAMBDA_BAND_LOW < LAMBDA_BAND_HIGH！" << endl;
					//return -1;
					businlog_error_assert(false, "Error: The bandpass wavelength must satisfy LAMBDA_BAND_LOW < LAMBDA_BAND_HIGH!");
				}
				const int maxImgRows = (OuterRadius - InnerRadius) / StepSize + 1; //径向扫描范围对应的最大行数
				//std::cout << __FUNCTION__ << " | maxImgRows:" << maxImgRows << std::endl;
				// 生成每行的系数
				std::vector<double> rows_coefficients(maxImgRows); //rows_coefficients[i]表示行i的系数 （元素从小到大排列）
				double outer_perimeter = 2 * CV_PI * OuterRadius;  //晶圆最外围周长
				double inner_perimeter = 2 * CV_PI * InnerRadius; //晶圆最内围周长
				double step = (outer_perimeter - inner_perimeter) / maxImgRows; //
				const double Scale = 1.0/*5.0*/; //值越大，目标越明显，同时可能引入假目标。周向采样点数为16384时，取2.5，或为32768时，取5.0，效果更明显，但为了和Candela61的结果相近，故使用1.0。
#pragma omp parallel for num_threads(m_max_threads)
				for (int y = 0; y < maxImgRows; y++)
				{
					rows_coefficients[static_cast<size_t>(maxImgRows - 1) - y] = Scale / (inner_perimeter + y * step);
				}
				if (srcImg.rows != M_ROW_COUNT || srcImg.cols != N_POINT_PER_ROW)
				{
					businlog_error_assert(false, "Incorrect input image size");
				}
#ifdef OUTPUT_DEBUG_INFO
				// 打印当前使用参数说明
				std::cout << "========== 当前滤波配置 ==========" << std::endl;
				std::cout << "图像尺寸：" << N_POINT_PER_ROW << " × " << M_ROW_COUNT << std::endl;
				if (filterType == FILTER_GAUSS)
				{
					std::cout << "滤波类型：高斯滤波 | ";
					if (filterMode == MODE_LOW || filterMode == MODE_HIGH)
					{
						std::cout << "模式：" << (filterMode == MODE_LOW ? "低通" : "高通") << std::endl;
						if (filterMode == MODE_LOW)
							std::cout << "使用波长参数 LAMBDA_BAND_LOW = " << LAMBDA_BAND_LOW << std::endl;
						if (filterMode == MODE_HIGH)
							std::cout << "使用波长参数 LAMBDA_BAND_HIGH = " << LAMBDA_BAND_HIGH << std::endl;
					}
					else
					{
						std::cout << "模式：带通" << std::endl;
						std::cout << "使用带通波长区间 [" << LAMBDA_BAND_LOW << "," << LAMBDA_BAND_HIGH << "]" << std::endl;
					}
				}
				else if (filterType == FILTER_UNIFORM)
				{
					std::cout << "滤波类型：Uniform矩形滤波 | ";
					if (filterMode == MODE_LOW || filterMode == MODE_HIGH)
					{
						std::cout << "模式：" << (filterMode == MODE_LOW ? "低通" : "高通") << std::endl;
						if (filterMode == MODE_LOW)
							std::cout << "使用波长参数 LAMBDA_BAND_LOW = " << LAMBDA_BAND_LOW << std::endl;
						if (filterMode == MODE_HIGH)
							std::cout << "使用波长参数 LAMBDA_BAND_HIGH = " << LAMBDA_BAND_HIGH << std::endl;
					}
					else
					{
						std::cout << "模式：带通" << std::endl;
						std::cout << "使用带通波长区间 [" << LAMBDA_BAND_LOW << "," << LAMBDA_BAND_HIGH << "]" << std::endl;
					}
				}
				else if (filterType == FilterType::FILTER_NONE) {
					std::cout << "filterType is None" << std::endl;
				}
				std::cout << "==================================" << std::endl;
#endif // OUTPUT_DEBUG_INFO
				std::vector<std::vector<double>> all_float_data(M_ROW_COUNT);

#pragma omp parallel for num_threads(m_max_threads)
				for (int y = 0; y < M_ROW_COUNT; y++)
				{
					int tid = omp_get_thread_num();
					auto& thread_res = m_thread_resources[tid];
					double dx = rows_coefficients[y];
					processSingleRow(y, srcImg, dx, filterType, filterMode, all_float_data[y], thread_res);
				}
				//std::cout << "频域滤波完成，转换16位图像..." << std::endl;
				cv::Mat img_out_u16;
				floatToUint16(all_float_data, img_out_u16);
				dstImg = img_out_u16;
				return 0;
			}

			// 外部读取行列接口
			inline int getRowPointNum() const { return N_POINT_PER_ROW; }
			inline int getTotalRowNum() const { return M_ROW_COUNT; }
		};
	}//nsCutOffFilter
}//nsLSR20
//函数原型定义