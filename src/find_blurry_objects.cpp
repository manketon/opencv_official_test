#include <opencv2/core/core.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
namespace fs = std::filesystem;

// 依据 MATLAB 脚本的 processSubImage 功能：输入单个子图（单通道，uint16/uint8），
// window_size 为列方向平滑窗口（纵向滤波），scale 为缩放比例（如 1.0/8）
static cv::Mat processSubImage(const cv::Mat& img_in, int window_size, double scale)
{
    // 转为 double（CV_64F）
    cv::Mat img;
    img_in.convertTo(img, CV_64F);

    int m = img.rows;
    int n = img.cols;

    // 列均值剖面 (m x 1)：每行（沿列方向）取平均 -> MATLAB: mean(img_gray,2)
    cv::Mat profile_col;
    cv::reduce(img, profile_col, 1, cv::REDUCE_AVG); // m x 1, CV_64F
    double mean_profile_col = cv::mean(profile_col)[0];
    if (mean_profile_col == 0) mean_profile_col = 1.0;
    profile_col /= mean_profile_col;

    // 行均值剖面 (1 x n)：每列（沿行方向）取平均 -> MATLAB: mean(img_gray,1)
    cv::Mat profile_row;
    cv::reduce(img, profile_row, 0, cv::REDUCE_AVG); // 1 x n, CV_64F
    double mean_profile_row = cv::mean(profile_row)[0];
    if (mean_profile_row == 0) mean_profile_row = 1.0;
    profile_row /= mean_profile_row; // still 1 x n

    // 先除以列均值（按行除），再除以行均值（按列除）
    cv::Mat colMat, rowMat;
    cv::repeat(profile_col, 1, n, colMat); // m x n
    cv::repeat(profile_row, m, 1, rowMat); // m x n

    cv::Mat img_correct = img.clone();
    img_correct = img_correct / colMat;
    img_correct = img_correct / rowMat;

    // 列方向均值平滑（使用 1D 垂直核） —— kernel size = window_size x 1
    cv::Mat kernel = cv::Mat::ones(window_size, 1, CV_64F) / static_cast<double>(window_size);
    cv::Mat img_smooth;
    cv::filter2D(img_correct, img_smooth, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

    // ---- 修改点：显式按 MATLAB 风格计算目标尺寸（向上取整），以避免 cv::resize 的四舍五入差异 ----
    int new_rows = static_cast<int>(std::ceil(m * scale));
    int new_cols = static_cast<int>(std::ceil(n * scale));
    // 保证至少为 1
    new_rows = std::max(1, new_rows);
    new_cols = std::max(1, new_cols);
    cv::Mat img_down;
    cv::resize(img_smooth, img_down, cv::Size(new_cols, new_rows), 0.0, 0.0, cv::INTER_CUBIC);

    // 裁剪到 [0,65535] 并转换为 uint16
    cv::Mat tmp;
    cv::max(img_down, 0.0, tmp);
    cv::min(tmp, 65535.0, tmp);
    cv::Mat img_down_uint16;
    tmp.convertTo(img_down_uint16, CV_16U);

    return img_down_uint16;
}

int test_find_blurry_objects()
{
    // ========== 参数（与 MATLAB 中对应） ==========
    string big_image_path = R"(C:\Users\LM\Desktop\1\5B_WD FA Sample 5B_01_1_QZrO_IPP.png)";
    string subimg_dir = "SubImages";
    string result_dir = "ProcessedResults";
    const int subimg_w = 2048;
    const int subimg_h = 482;

    // 物理参数
    const double angle_total = 360.0;
    const double radius_start = 15500.0; // μm
    const double radius_end = 47000.0; // μm
    const double radius_span = radius_end - radius_start;
    // 下采样比例与平滑窗口（与 MATLAB 保持一致）
    const int window_size = 60;
    const double scale = 1.0 / 8.0;

    // 创建输出文件夹
    if (!fs::exists(subimg_dir)) {
        fs::create_directories(subimg_dir);
    }
    if (!fs::exists(result_dir)) {
        fs::create_directories(result_dir);
    }

    // 读取大图（保留原始位深）
    cout << "读取大图...\n";
    cv::Mat I = cv::imread(big_image_path, cv::IMREAD_UNCHANGED);
    if (I.empty()) {
        cerr << "无法读取图像: " << big_image_path << endl;
        return -1;
    }

    // 如果是彩色，转为灰度（支持 8U/16U 彩色转灰）
    if (I.channels() == 3 || I.channels() == 4) {
        cv::Mat gray;
        if (I.depth() == CV_8U) cv::cvtColor(I, gray, cv::COLOR_BGR2GRAY);
        else cv::cvtColor(I, gray, cv::COLOR_BGR2GRAY); // 对 16U 也可直接使用
        I = gray;
    }

    int rows = I.rows;
    int cols = I.cols;
    cout << "图像尺寸: " << rows << " x " << cols << endl;

    if (cols % subimg_w != 0 || rows % subimg_h != 0) {
        cerr << "原图尺寸不能被子图尺寸整除，请检查参数！" << endl;
        return -2;
    }

    int num_horiz = cols / subimg_w; // 列数
    int num_vert = rows / subimg_h; // 行数

    double pixel_size_radial = radius_span / static_cast<double>(rows);
    double sub_angle_range = static_cast<double>(subimg_w) / static_cast<double>(cols) * angle_total;
    double sub_radius_range = static_cast<double>(subimg_h) * pixel_size_radial;

    // 打开 CSV 映射表
    std::ofstream fid("subimage_mapping.csv");
    fid << "序号,角度索引,半径索引,角度范围(deg),半径范围(um)\n";

    // 切割并保存子图（物理坐标命名）
    cout << "开始切割子图...\n";
    int count = 0;
    for (int i = 1; i <= num_vert; ++i) { // MATLAB 从 1 到 num_vert
        for (int j = 1; j <= num_horiz; ++j) {
            ++count;
            int row_start = (i - 1) * subimg_h;
            int row_end = i * subimg_h; // exclusive for cv::Range

            int col_start = (j - 1) * subimg_w;
            int col_end = j * subimg_w;

            cv::Mat subImg = I(cv::Range(row_start, row_end), cv::Range(col_start, col_end)).clone();

            int angle_idx = j - 1;
            int radius_idx = i - 1;

            double angle_start_val = angle_idx * sub_angle_range;
            double radius_start_val = radius_end - (i - 1) * sub_radius_range;

            char filename[256];
            std::snprintf(filename, sizeof(filename), "%04d_ang%.1f_r%.0f.png",
                count, angle_start_val, radius_start_val);
            fs::path fullname = fs::path(subimg_dir) / filename;

            // 保存子图（保持原始深度）
            if (!cv::imwrite(fullname.string(), subImg)) {
                cerr << "保存子图失败: " << fullname.string() << endl;
            }

            double angle_min = angle_start_val;
            double angle_max = (angle_idx + 1) * sub_angle_range;
            double radius_max = radius_end - (i - 1) * sub_radius_range;
            double radius_min = radius_end - i * sub_radius_range;

            // 写入映射表
            char linebuf[512];
            std::snprintf(linebuf, sizeof(linebuf), "%d,%d,%d,%.4f-%.4f,%.2f-%.2f\n",
                count, angle_idx, radius_idx,
                angle_min, angle_max, radius_min, radius_max);
            fid << linebuf;
        }
    }
    fid.close();
    cout << "切割完成，共 " << count << " 个子图。\n";

    // 逐子图处理并保存结果
    cout << "开始处理子图...\n";
    for (int i = 1; i <= num_vert; ++i) {
        for (int j = 1; j <= num_horiz; ++j) {
            int idx = (i - 1) * num_horiz + j;
            int angle_idx = j - 1;
            int radius_idx = i - 1;
            double angle_start_val = angle_idx * sub_angle_range;
            double radius_start_val = radius_end - (i - 1) * sub_radius_range;

            char inname[256];
            std::snprintf(inname, sizeof(inname), "%04d_ang%.1f_r%.0f.png",
                idx, angle_start_val, radius_start_val);
            fs::path infile = fs::path(subimg_dir) / inname;

            cv::Mat img = cv::imread(infile.string(), cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                cerr << "无法读取子图: " << infile.string() << "，跳过。\n";
                continue;
            }

            cv::Mat img_down_uint16 = processSubImage(img, window_size, scale);

            char outname[256];
            std::snprintf(outname, sizeof(outname), "proc_%04d_ang%.1f_r%.0f.png",
                idx, angle_start_val, radius_start_val);
            fs::path outfile = fs::path(result_dir) / outname;

            if (!cv::imwrite(outfile.string(), img_down_uint16)) {
                cerr << "保存处理结果失败: " << outfile.string() << endl;
            }
        }
    }
    cout << "全部处理完成！\n";
    return 0;
}