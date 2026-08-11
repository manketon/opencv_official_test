#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846

// 滤波类型
enum FilterType
{
	FILTER_GAUSS,  //高斯
	FILTER_UNIFORM  //uniform
};
// 滤波模式
enum FilterMode
{
	MODE_LOW, //低通
	MODE_HIGH, //高通
	MODE_BAND  //带通
};

class RowFFTMultiFilter
{
private:
	// 动态输入参数（构造传入）
	int N_POINT_PER_ROW;  // 每行采样点数
	int M_ROW_COUNT;  // 总行数
	double LAMBDA_BAND_LOW; //低通波长，使用Uniform的时候使用
	double LAMBDA_BAND_HIGH; //高通波长，使用Uniform的时候使用

	// FFTW 资源
	fftw_plan fft_plan;
	fftw_plan ifft_plan;
	double* in_real_buf;
	fftw_complex* fft_cplx_buf;

	/**
	 * 1D 频谱中心化 fftshift：低频挪到数组中间
	 * @param cplx 复数频域数组
	 */
	void fftShift1D(fftw_complex* cplx)
	{
		int half = N_POINT_PER_ROW / 2;
		vector<fftw_complex> temp(half);
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

	// ===================== 新增：打印一段复数频谱（只打印局部，防止刷屏） =====================
	void printFFTComplexSlice(fftw_complex* cplx, int start, int end, string desc)
	{
		cout << "\n===== " << desc << " =====" << endl;
		cout << setprecision(6) << fixed;
		for (int k = start; k <= end; k++)
		{
			double re = cplx[k][0];
			double im = cplx[k][1];
			double amp = sqrt(re * re + im * im);
			cout << "k=" << setw(5)
				<< "  Re=" << setw(10) << re
				<< "  Im=" << setw(10) << im
				<< "  Amp=" << setw(10) << amp << endl;
		}
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
			if (fMode == MODE_BAND)
			{
				cerr << "错误：高斯不支持带通模式" << endl;
				return 1.0;
			}
			double f0 = 0.0;
			if (fMode == MODE_LOW)
				f0 = 1.0 / LAMBDA_BAND_LOW;
			if (fMode == MODE_HIGH)
				f0 = 1.0 / LAMBDA_BAND_HIGH;
			double sqrt_2ln2 = sqrt(2 * log(2.0));
			double SIGMA_F = f0 / sqrt_2ln2;
			double H_low = exp(-(f * f) / (2.0 * SIGMA_F * SIGMA_F));
			return (fMode == MODE_LOW) ? H_low : (1.0 - H_low);
		}
		// Uniform 滤波
		else
		{
			// 低通/高通：
			if (fMode == MODE_LOW || fMode == MODE_HIGH)
			{
				double f0 = 0.0;
				if (fMode == MODE_LOW)
					f0 = 1.0 / LAMBDA_BAND_LOW;
				if (fMode == MODE_HIGH)
					f0 = 1.0 / LAMBDA_BAND_HIGH;
				double abs_f = fabs(f);
				if (fMode == MODE_LOW)
					return (abs_f <= f0) ? 1.0 : 0.0;
				else
					return (abs_f > f0) ? 1.0 : 0.0;
			}
			// 带通：使用band区间波长
			else if (fMode == MODE_BAND)
			{
				double f_band_low = 1.0 / LAMBDA_BAND_HIGH;
				double f_band_high = 1.0 / LAMBDA_BAND_LOW;
				double abs_f = fabs(f);
				return (abs_f >= f_band_low && abs_f <= f_band_high) ? 1.0 : 0.0;
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
	void processSingleRow(const Mat& row_in, double dx_step, FilterType filterType, FilterMode filterMode, vector<double>& row_out)
	{
		// 1. 载入原始行到FFTW实数缓冲区
		for (int x = 0; x < N_POINT_PER_ROW; x++)
		{
			in_real_buf[x] = static_cast<double>(row_in.at<ushort>(0, x));
		}
		// 2. 正向傅里叶变换
		fftw_execute(fft_plan);
		// 3. 频谱中心化 fftshift
		fftShift1D(fft_cplx_buf);

		int half = N_POINT_PER_ROW / 2;
		for (int k = 0; k < N_POINT_PER_ROW; k++)
		{
			// 中心化后频率偏移：中心k=half对应f=0
			int delta_k = k - half;
			double f = delta_k * dx_step; // 真实物理频率(正负)
			double H = calcTransmittance(f, filterType, filterMode);
			// 频域复数乘以高斯透过率
			double re = fft_cplx_buf[k][0];
			double im = fft_cplx_buf[k][1];
			fft_cplx_buf[k][0] = re * H;
			fft_cplx_buf[k][1] = im * H;
		}
		// 5. 逆中心化 ifftshift
		ifftShift1D(fft_cplx_buf);
		// 6. 逆傅里叶变换
		fftw_execute(ifft_plan);
		// 7. IFFT归一化，提取实部输出
		row_out.resize(N_POINT_PER_ROW);
		double scale = 1.0 / N_POINT_PER_ROW;
		for (int x = 0; x < N_POINT_PER_ROW; x++)
		{
			double temp_x_new = in_real_buf[x] * scale;
			row_out[x] = in_real_buf[x] * scale;
		}
	}

	/**
	 * 浮点二维数组全局归一化映射到 uint16 [0,4095]
	 */
	void floatToUint16(const vector<vector<double>>& float_buf, Mat& dst_u16)
	{
		dst_u16.create(M_ROW_COUNT, N_POINT_PER_ROW, CV_16UC1);
		double global_min = 1e15;
		double global_max = -1e15;

		for (int y = 0; y < M_ROW_COUNT; y++)
		{
			const auto& row = float_buf[y];
			double rmin = *min_element(row.begin(), row.end());
			double rmax = *max_element(row.begin(), row.end());
			if (rmin < global_min) global_min = rmin;
			if (rmax > global_max) global_max = rmax;
		}

		double range = fabs(global_max - global_min) < 1e-9 ? 1.0 : (global_max - global_min);
		for (int y = 0; y < M_ROW_COUNT; y++)
		{
			const auto& row = float_buf[y];
			for (int x = 0; x < N_POINT_PER_ROW; x++)
			{
				double val = row[x];
				double norm = (val - global_min) / range * 4095.0;
				dst_u16.at<ushort>(y, x) = static_cast<ushort>(round(norm));
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
		LAMBDA_BAND_HIGH(bandHigh)
	{
		// ========== 参数合法性校验 ==========
		if (N_POINT_PER_ROW <= 0)
			throw invalid_argument("每行采样点数 N_POINT_PER_ROW 必须大于0");
		if (M_ROW_COUNT <= 0)
			throw invalid_argument("总行数 M_ROW_COUNT 必须大于0");
		if (LAMBDA_BAND_LOW <= 0.0)
			throw invalid_argument("LAMBDA_BAND_LOW 带通下限波长必须大于0");
		if (LAMBDA_BAND_HIGH <= 0.0)
			throw invalid_argument("LAMBDA_BAND_HIGH 带通上限波长必须大于0");
		if (LAMBDA_BAND_LOW >= LAMBDA_BAND_HIGH)
			throw invalid_argument("带通波长必须满足 LAMBDA_BAND_LOW < LAMBDA_BAND_HIGH");

		// 分配FFTW缓冲区，初始化FFTW资源
		in_real_buf = (double*)fftw_malloc(sizeof(double) * N_POINT_PER_ROW);
		fft_cplx_buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N_POINT_PER_ROW);
		fft_plan = fftw_plan_dft_r2c_1d(N_POINT_PER_ROW, in_real_buf, fft_cplx_buf, FFTW_MEASURE);
		ifft_plan = fftw_plan_dft_c2r_1d(N_POINT_PER_ROW, fft_cplx_buf, in_real_buf, FFTW_MEASURE);
	}

	// 析构自动释放FFTW内存
	~RowFFTMultiFilter()
	{
		fftw_destroy_plan(fft_plan);
		fftw_destroy_plan(ifft_plan);
		fftw_free(in_real_buf);
		fftw_free(fft_cplx_buf);
	}

	/**
	 * 主处理接口
	 */
	int runProcess(const string& inputPath,
		const string& outputPath,
		vector<double>& dx_list,
		FilterType filterType,
		FilterMode filterMode)
	{
		// 高斯禁止带通
		if (filterType == FILTER_GAUSS && filterMode == MODE_BAND)
		{
			cerr << "错误：高斯滤波不支持带通模式！" << endl;
			return -1;
		}
		// dx数组长度校验
		if ((int)dx_list.size() != M_ROW_COUNT)
		{
			cerr << "错误：dx步距数组长度必须等于图像总行数" << endl;
			return -1;
		}

		Mat img_src = imread(inputPath, IMREAD_UNCHANGED);
		if (img_src.empty())
		{
			cerr << "错误：读取图像失败！" << endl;
			return -1;
		}
		if (img_src.rows != M_ROW_COUNT || img_src.cols != N_POINT_PER_ROW)
		{
			cerr << "错误：图像尺寸与初始化行列参数不匹配！" << endl;
			return -1;
		}

		// 打印当前使用参数说明
		cout << "========== 当前滤波配置 ==========" << endl;
		cout << "图像尺寸：" << N_POINT_PER_ROW << " × " << M_ROW_COUNT << endl;
		if (filterType == FILTER_GAUSS)
		{
			cout << "滤波类型：高斯滤波 | 模式：" << (filterMode == MODE_LOW ? "低通" : "高通") << endl;
			if (filterMode == MODE_LOW)
				cout << "使用波长参数 LAMBDA_BAND_LOW = " << LAMBDA_BAND_LOW << endl;
			if (filterMode == MODE_HIGH)
				cout << "使用波长参数 LAMBDA_BAND_HIGH = " << LAMBDA_BAND_HIGH << endl;
		}
		else
		{
			cout << "滤波类型：Uniform矩形滤波 | ";
			if (filterMode == MODE_LOW || filterMode == MODE_HIGH)
			{
				cout << "模式：" << (filterMode == MODE_LOW ? "低通" : "高通") << endl;
				if (filterMode == MODE_LOW)
					cout << "使用波长参数 LAMBDA_BAND_LOW = " << LAMBDA_BAND_LOW << endl;
				if (filterMode == MODE_HIGH)
					cout << "使用波长参数 LAMBDA_BAND_HIGH = " << LAMBDA_BAND_HIGH << endl;
			}
			else
			{
				cout << "模式：带通" << endl;
				cout << "使用带通波长区间 [" << LAMBDA_BAND_LOW << "," << LAMBDA_BAND_HIGH << "]" << endl;
			}
		}
		cout << "==================================" << endl;

		vector<vector<double>> all_float_data(M_ROW_COUNT);
		for (int y = 0; y < M_ROW_COUNT; y++)
		{
			Mat single_row = img_src.row(y);
			double dx = dx_list[y];
			processSingleRow(single_row, dx, filterType, filterMode, all_float_data[y]);
		}

		cout << "频域滤波完成，转换16位图像..." << endl;
		Mat img_out_u16;
		floatToUint16(all_float_data, img_out_u16);

		imwrite(outputPath, img_out_u16);
		cout << "输出图像保存成功：" << outputPath << endl;
		return 0;
	}

	// 外部读取行列接口
	inline int getRowPointNum() const { return N_POINT_PER_ROW; }
	inline int getTotalRowNum() const { return M_ROW_COUNT; }
};

int test_wrinkle()
{
	try
	{
		// 图像尺寸
		const int W = 32768;
		const int H = 8063;
		// 生成每行扫描步距dx数组
		vector<double> dx_list(H);
		double out_max = 2 * M_PI * 48000.0;  //晶圆最外围尺寸半径
		double inner_min = 2 * M_PI * 16000.0; //晶圆最内围尺寸半径
		double step = (out_max - inner_min) / H; //步长
		for (int y = 0; y < H; y++)
		{
			dx_list[y] = 1.0 / (inner_min + y * step);
		}
		sort(dx_list.begin(), dx_list.end());

		string in_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\1.png)";
		//string out_path = "result_band.png";
		string out_path = R"(G:\wafer_images\wrinkleImgs\dataForWrinkle\result_GAUSS_HIGH.png)";
		//string out_path = "result_Uniform_band.png";

		// ========== 切换此处参数即可更换滤波 ==========
		// 示例1：高斯低通
		//FilterType fType = FILTER_GAUSS;
		//FilterMode fMode = MODE_LOW;

		//// 示例2：高斯高通
		FilterType fType = FILTER_GAUSS;
		FilterMode fMode = MODE_HIGH;

		// 示例3：Uniform低通
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_LOW;

		//// 示例4：Uniform高通
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_HIGH;

		//// 示例5：Uniform带通（波长50~150）
		//FilterType fType = FILTER_UNIFORM;
		//FilterMode fMode = MODE_BAND;

		// 波长参数（可自行修改）
		double bandLowLam = 100.0; //如果没有选中Uniform带通模式，则不使用这两个参数。默认参数设为1。
		double bandHighLam = 250.0; //如果没有选中Uniform带通模式，则不使用这两个参数。默认参数设为1000。

		// 构造对象，传入全部动态参数（内部自动校验）
		//RowFFTMultiFilter filter(W, H, targetWave, bandLowLam, bandHighLam);
		RowFFTMultiFilter filter(W, H, bandLowLam, bandHighLam);

		int ret = filter.runProcess(in_path, out_path, dx_list, fType, fMode);
		if (ret != 0)
		{
			cerr << "图像处理失败！" << endl;
			return -1;
		}
	}
	catch (const invalid_argument& e)
	{
		cerr << "参数校验失败：" << e.what() << endl;
		return -1;
	}
	catch (...)
	{
		cerr << "程序发生未知异常！" << endl;
		return -1;
	}
	return 0;
}