#include "hough.hpp"
#include <opencv2\core\opencl\ocl_defs.hpp>
#include <opencv2\core\ocl.hpp>
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <algorithm>
#include <iterator>
using namespace cv;
namespace YRP
{
	static inline int
		computeNumangle(double min_theta, double max_theta, double theta_step)
	{
		int numangle = cvFloor((max_theta - min_theta) / theta_step) + 1;
		// If the distance between the first angle and the last angle is
		// approximately equal to pi, then the last angle will be removed
		// in order to prevent a line to be detected twice.
		if (numangle > 1 && fabs(CV_PI - (numangle - 1) * theta_step) < theta_step / 2)
			--numangle;
		return numangle;
	}

#ifdef HAVE_OPENCL

#define OCL_MAX_LINES 4096

	static bool ocl_makePointsList(InputArray _src, OutputArray _pointsList, InputOutputArray _counters)
	{
		UMat src = _src.getUMat();
		_pointsList.create(1, (int)src.total(), CV_32SC1);
		UMat pointsList = _pointsList.getUMat();
		UMat counters = _counters.getUMat();
		ocl::Device dev = ocl::Device::getDefault();

		const int pixPerWI = 16;
		int workgroup_size = min((int)dev.maxWorkGroupSize(), (src.cols + pixPerWI - 1) / pixPerWI);
		ocl::Kernel pointListKernel("make_point_list", ocl::imgproc::hough_lines_oclsrc,
			format("-D MAKE_POINTS_LIST -D GROUP_SIZE=%d -D LOCAL_SIZE=%d", workgroup_size, src.cols));
		if (pointListKernel.empty())
			return false;

		pointListKernel.args(ocl::KernelArg::ReadOnly(src), ocl::KernelArg::WriteOnlyNoSize(pointsList),
			ocl::KernelArg::PtrWriteOnly(counters));

		size_t localThreads[2] = { (size_t)workgroup_size, 1 };
		size_t globalThreads[2] = { (size_t)workgroup_size, (size_t)src.rows };

		return pointListKernel.run(2, globalThreads, localThreads, false);
	}

	static bool ocl_fillAccum(InputArray _pointsList, OutputArray _accum, int total_points, double rho, double theta, int numrho, int numangle)
	{
		UMat pointsList = _pointsList.getUMat();
		_accum.create(numangle + 2, numrho + 2, CV_32SC1);
		UMat accum = _accum.getUMat();
		ocl::Device dev = ocl::Device::getDefault();

		float irho = (float)(1 / rho);
		int workgroup_size = min((int)dev.maxWorkGroupSize(), total_points);

		ocl::Kernel fillAccumKernel;
		size_t localThreads[2];
		size_t globalThreads[2];

		size_t local_memory_needed = (numrho + 2) * sizeof(int);
		if (local_memory_needed > dev.localMemSize())
		{
			accum.setTo(Scalar::all(0));
			fillAccumKernel.create("fill_accum_global", ocl::imgproc::hough_lines_oclsrc,
				format("-D FILL_ACCUM_GLOBAL"));
			if (fillAccumKernel.empty())
				return false;
			globalThreads[0] = workgroup_size; globalThreads[1] = numangle;
			fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
				total_points, irho, (float)theta, numrho, numangle);
			return fillAccumKernel.run(2, globalThreads, NULL, false);
		}
		else
		{
			fillAccumKernel.create("fill_accum_local", ocl::imgproc::hough_lines_oclsrc,
				format("-D FILL_ACCUM_LOCAL -D LOCAL_SIZE=%d -D BUFFER_SIZE=%d", workgroup_size, numrho + 2));
			if (fillAccumKernel.empty())
				return false;
			localThreads[0] = workgroup_size; localThreads[1] = 1;
			globalThreads[0] = workgroup_size; globalThreads[1] = numangle + 2;
			fillAccumKernel.args(ocl::KernelArg::ReadOnlyNoSize(pointsList), ocl::KernelArg::WriteOnlyNoSize(accum),
				total_points, irho, (float)theta, numrho, numangle);
			return fillAccumKernel.run(2, globalThreads, localThreads, false);
		}
	}

	static bool ocl_HoughLines(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
		double min_theta, double max_theta)
	{
		CV_Assert(_src.type() == CV_8UC1);

		if (max_theta < 0 || max_theta > CV_PI) {
			CV_Error(Error::StsBadArg, "max_theta must fall between 0 and pi");
		}
		if (min_theta < 0 || min_theta > max_theta) {
			CV_Error(Error::StsBadArg, "min_theta must fall between 0 and max_theta");
		}
		if (!(rho > 0 && theta > 0)) {
			CV_Error(Error::StsBadArg, "rho and theta must be greater 0");
		}

		UMat src = _src.getUMat();
		int numangle = computeNumangle(min_theta, max_theta, theta);
		int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

		UMat pointsList;
		UMat counters(1, 2, CV_32SC1, Scalar::all(0));

		if (!ocl_makePointsList(src, pointsList, counters))
			return false;

		int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
		if (total_points <= 0)
		{
			_lines.release();
			return true;
		}

		UMat accum;
		if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
			return false;

		const int pixPerWI = 8;
		ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
			format("-D GET_LINES"));
		if (getLinesKernel.empty())
			return false;

		int linesMax = threshold > 0 ? min(total_points * numangle / threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
		UMat lines(linesMax, 1, CV_32FC2);

		getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::WriteOnlyNoSize(lines),
			ocl::KernelArg::PtrWriteOnly(counters), linesMax, threshold, (float)rho, (float)theta);

		size_t globalThreads[2] = { ((size_t)numrho + pixPerWI - 1) / pixPerWI, (size_t)numangle };
		if (!getLinesKernel.run(2, globalThreads, NULL, false))
			return false;

		int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
		if (total_lines > 0)
			_lines.assign(lines.rowRange(Range(0, total_lines)));
		else
			_lines.release();
		return true;
	}

	static bool ocl_HoughLinesP(InputArray _src, OutputArray _lines, double rho, double theta, int threshold,
		double minLineLength, double maxGap)
	{
		CV_Assert(_src.type() == CV_8UC1);

		if (!(rho > 0 && theta > 0)) {
			CV_Error(Error::StsBadArg, "rho and theta must be greater 0");
		}

		UMat src = _src.getUMat();
		int numangle = computeNumangle(0.0, CV_PI, theta);
		int numrho = cvRound(((src.cols + src.rows) * 2 + 1) / rho);

		UMat pointsList;
		UMat counters(1, 2, CV_32SC1, Scalar::all(0));

		if (!ocl_makePointsList(src, pointsList, counters))
			return false;

		int total_points = counters.getMat(ACCESS_READ).at<int>(0, 0);
		if (total_points <= 0)
		{
			_lines.release();
			return true;
		}

		UMat accum;
		if (!ocl_fillAccum(pointsList, accum, total_points, rho, theta, numrho, numangle))
			return false;

		ocl::Kernel getLinesKernel("get_lines", ocl::imgproc::hough_lines_oclsrc,
			format("-D GET_LINES_PROBABOLISTIC"));
		if (getLinesKernel.empty())
			return false;

		int linesMax = threshold > 0 ? min(total_points * numangle / threshold, OCL_MAX_LINES) : OCL_MAX_LINES;
		UMat lines(linesMax, 1, CV_32SC4);

		getLinesKernel.args(ocl::KernelArg::ReadOnly(accum), ocl::KernelArg::ReadOnly(src),
			ocl::KernelArg::WriteOnlyNoSize(lines), ocl::KernelArg::PtrWriteOnly(counters),
			linesMax, threshold, (int)minLineLength, (int)maxGap, (float)rho, (float)theta);

		size_t globalThreads[2] = { (size_t)numrho, (size_t)numangle };
		if (!getLinesKernel.run(2, globalThreads, NULL, false))
			return false;

		int total_lines = min(counters.getMat(ACCESS_READ).at<int>(0, 1), linesMax);
		if (total_lines > 0)
			_lines.assign(lines.rowRange(Range(0, total_lines)));
		else
			_lines.release();

		return true;
	}

#endif /* HAVE_OPENCL */

	/****************************************************************************************\
	*                              Probabilistic Hough Transform                             *
	\****************************************************************************************/
	static void
	HoughLinesProbabilistic(Mat& image,
			float rho, float theta, int threshold,
			int lineLength, int lineGap,
			std::vector<Vec4i>& lines, int linesMax)
	{
		Point pt;
		float irho = 1 / rho;
		RNG rng((uint64)-1);

		CV_Assert(image.type() == CV_8UC1);

		int width = image.cols;
		int height = image.rows;

		int numangle = computeNumangle(0.0, CV_PI, theta);
		int numrho = cvRound(((width + height) * 2 + 1) / rho);

#if defined HAVE_IPP && IPP_VERSION_X100 >= 810 && !IPP_DISABLE_HOUGH
		CV_IPP_CHECK()
		{
			IppiSize srcSize = { width, height };
			IppPointPolar delta = { rho, theta };
			IppiHoughProbSpec* pSpec;
			int bufferSize, specSize;
			int ipp_linesMax = std::min(linesMax, numangle * numrho);
			int linesCount = 0;
			lines.resize(ipp_linesMax);
			IppStatus ok = ippiHoughProbLineGetSize_8u_C1R(srcSize, delta, &specSize, &bufferSize);
			Ipp8u* buffer = ippsMalloc_8u_L(bufferSize);
			pSpec = (IppiHoughProbSpec*)ippsMalloc_8u_L(specSize);
			if (ok >= 0) ok = ippiHoughProbLineInit_8u32f_C1R(srcSize, delta, ippAlgHintNone, pSpec);
			if (ok >= 0) { ok = CV_INSTRUMENT_FUN_IPP(ippiHoughProbLine_8u32f_C1R, image.data, (int)image.step, srcSize, threshold, lineLength, lineGap, (IppiPoint*)&lines[0], ipp_linesMax, &linesCount, buffer, pSpec); };

			ippsFree(pSpec);
			ippsFree(buffer);
			if (ok >= 0)
			{
				lines.resize(linesCount);
				CV_IMPL_ADD(CV_IMPL_IPP);
				return;
			}
			lines.clear();
			setIppErrorStatus();
		}
#endif

		Mat accum = Mat::zeros(numangle, numrho, CV_32SC1);
		Mat mask(height, width, CV_8UC1);
		std::vector<float> trigtab(numangle * 2);

		for (int n = 0; n < numangle; n++)
		{
			trigtab[n * 2] = (float)(cos((double)n * theta) * irho);
			trigtab[n * 2 + 1] = (float)(sin((double)n * theta) * irho);
		}
		const float* ttab = &trigtab[0];
		uchar* mdata0 = mask.ptr();
		std::vector<Point> nzloc;

		// stage 1. collect non-zero image points
		for (pt.y = 0; pt.y < height; pt.y++)
		{
			const uchar* data = image.ptr(pt.y);
			uchar* mdata = mask.ptr(pt.y);
			for (pt.x = 0; pt.x < width; pt.x++)
			{
				if (data[pt.x])
				{
					mdata[pt.x] = (uchar)1;
					nzloc.push_back(pt);
				}
				else
					mdata[pt.x] = 0;
			}
		}

		int count = (int)nzloc.size();

		// stage 2. process all the points in random order
		for (; count > 0; count--)
		{
			// choose random point out of the remaining ones
			int idx = rng.uniform(0, count);
			int max_val = threshold - 1, max_n = 0;
			Point point = nzloc[idx];
			Point line_end[2];
			float a, b;
			int* adata = accum.ptr<int>();
			int i = point.y, j = point.x, k, xflag;
			int64_t x0, y0, dx0, dy0;
			int good_line;
			const int shift = 16;

			// "remove" it by overriding it with the last element
			nzloc[idx] = nzloc[count - 1];

			// check if it has been excluded already (i.e. belongs to some other line)
			if (!mdata0[i * width + j])
				continue;

			// update accumulator, find the most probable line
			for (int n = 0; n < numangle; n++, adata += numrho)
			{
				int r = cvRound(j * ttab[n * 2] + i * ttab[n * 2 + 1]);
				r += (numrho - 1) / 2;
				int val = ++adata[r];
				if (max_val < val)
				{
					max_val = val;
					max_n = n;
				}
			}

			// if it is too "weak" candidate, continue with another point
			if (max_val < threshold)
				continue;

			// from the current point walk in each direction
			// along the found line and extract the line segment
			a = -ttab[max_n * 2 + 1];
			b = ttab[max_n * 2];
			x0 = j;
			y0 = i;
			if (fabs(a) > fabs(b))
			{
				xflag = 1;
				dx0 = a > 0 ? 1 : -1;
				dy0 = cvRound(b * (1 << shift) / fabs(a));
				y0 = (y0 << shift) + (1 << (shift - 1));
			}
			else
			{
				xflag = 0;
				dy0 = b > 0 ? 1 : -1;
				dx0 = cvRound(a * (1 << shift) / fabs(b));
				x0 = (x0 << shift) + (1 << (shift - 1));
			}

			for (k = 0; k < 2; k++)
			{
				int gap = 0;
				int64_t x = x0, y = y0, dx = dx0, dy = dy0;
				if (k > 0)
					dx = -dx, dy = -dy;

				// walk along the line using fixed-point arithmetic,
				// stop at the image border or in case of too big gap
				for (;; x += dx, y += dy)
				{
					uchar* mdata;
					int64_t i1, j1;

					if (xflag)
					{
						j1 = x;
						i1 = y >> shift;
					}
					else
					{
						j1 = x >> shift;
						i1 = y;
					}

					if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
						break;

					mdata = mdata0 + i1 * width + j1;

					// for each non-zero point:
					//    update line end,
					//    clear the mask element
					//    reset the gap
					if (*mdata)
					{
						gap = 0;
						line_end[k].y = static_cast<int>(i1);
						line_end[k].x = static_cast<int>(j1);
					}
					else if (++gap > lineGap)
						break;
				}
			}

			good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
				std::abs(line_end[1].y - line_end[0].y) >= lineLength;

			for (k = 0; k < 2; k++)
			{
				int64_t x = x0, y = y0, dx = dx0, dy = dy0;

				if (k > 0)
					dx = -dx, dy = -dy;

				// walk along the line using fixed-point arithmetic,
				// stop at the image border or in case of too big gap
				for (;; x += dx, y += dy)
				{
					uchar* mdata;
					int64_t i1, j1;

					if (xflag)
					{
						j1 = x;
						i1 = y >> shift;
					}
					else
					{
						j1 = x >> shift;
						i1 = y;
					}

					mdata = mdata0 + i1 * width + j1;

					// for each non-zero point:
					//    update line end,
					//    clear the mask element
					//    reset the gap
					if (*mdata)
					{
						if (good_line)
						{
							adata = accum.ptr<int>();
							for (int n = 0; n < numangle; n++, adata += numrho)
							{
								int r = cvRound(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1]);
								r += (numrho - 1) / 2;
								adata[r]--;
							}
						}
						*mdata = 0;
					}

					if (i1 == line_end[k].y && j1 == line_end[k].x)
						break;
				}
			}

			if (good_line)
			{
				Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
				lines.push_back(lr);
				if ((int)lines.size() >= linesMax)
					return;
			}
		}
	}

	void HoughLinesP(cv::InputArray _image, cv::OutputArray _lines, double rho, double theta, int threshold, double minLineLength, double maxGap)
	{
		CV_OCL_RUN(_image.isUMat() && _lines.isUMat(),
			ocl_HoughLinesP(_image, _lines, rho, theta, threshold, minLineLength, maxGap));

		Mat image = _image.getMat();
		std::vector<Vec4i> lines;
		HoughLinesProbabilistic(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX);
		Mat(lines).copyTo(_lines);
	}

}

