#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

// 自定义 tEXt 键值对
using TextMap = std::map<std::string, std::string>;

// 写入带自定义tEXt的16UC1(单通道uint16)PNG
bool savePng16UC1(const std::string& path,
	int width, int height,
	const uint16_t* data,
	const TextMap& txtMeta)
{
	FILE* fp = fopen(path.c_str(), "wb");
	if (!fp) return false;

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!png_ptr) { fclose(fp); return false; }

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr, nullptr);
		fclose(fp);
		return false;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(fp);
		return false;
	}

	png_init_io(png_ptr, fp);

	// ========== 1. 设置图像参数：16bit, GRAY(单通道)=16UC1 ==========
	const int bit_depth = 16;
	const int color_type = PNG_COLOR_TYPE_GRAY; // 单通道灰度
	png_set_IHDR(png_ptr, info_ptr,
		width, height,
		bit_depth, color_type,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);

	// ========== 2. 批量添加 tEXt 元数据（核心：自定义头部文本） ==========
	if (!txtMeta.empty())
	{
		std::vector<png_text> text_list;
		for (auto& kv : txtMeta)
		{
			png_text tx{};
			tx.compression = PNG_TEXT_COMPRESSION_NONE; // tEXt 无压缩
			tx.key = const_cast<char*>(kv.first.c_str());
			tx.text = const_cast<char*>(kv.second.c_str());
			tx.text_length = kv.second.size();
			text_list.push_back(tx);
		}
		png_set_text(png_ptr, info_ptr, text_list.data(), text_list.size());
	}

	// 写入 IHDR + tEXt 块（此时tEXt自动插在IHDR后、IDAT前）
	png_write_info(png_ptr, info_ptr);

	// ========== 3. 组装行指针，写入像素数据(uint16小端/大端说明) ==========
	// libpng默认网络字节序(BE大端)；你的原始数据如果是CPU小端(x86)需要字节翻转
	std::vector<png_bytep> row_ptrs(height);
	for (int y = 0; y < height; y++)
	{
		row_ptrs[y] = reinterpret_cast<png_bytep>(const_cast<uint16_t*>(data) + y * width);
	}
	png_write_image(png_ptr, row_ptrs.data());
	png_write_end(png_ptr, info_ptr);

	// 释放资源
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
	return true;
}

int test_write_info_to_png()
{
	// 1. 自定义元数据（和你图片的tEXt字段一致）
	TextMap meta{
		{"tEXtFileName",        "SIC-EPI-xxx.png"},
		{"tEXtPMTvoltage",      "643.2374866074638"},
		{"tEXtWidth",           "2079"},
		{"tEXtHeight",          "351"},
		{"tEXtWaferId",         "HY1320382"}
	};

	// 2. 构造测试 16UC1 图像数据 H=351 W=2079
	const int W = 2079;
	const int H = 351;
	std::vector<uint16_t> buf(W * H, 1000); // 全填充灰度值

	// 3. 保存PNG
	bool ok = savePng16UC1("out_16UC1_meta.png", W, H, buf.data(), meta);
	if (ok) printf("save success\n");
	else    printf("save fail\n");

	return 0;
}