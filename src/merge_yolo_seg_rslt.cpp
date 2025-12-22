#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <iostream>

// 哈希函数：用于unordered_map的键（桶坐标对）
struct PairHash {
	template <typename T1, typename T2>
	std::size_t operator()(const std::pair<T1, T2>& p) const {
		auto h1 = std::hash<T1>{}(p.first);
		auto h2 = std::hash<T2>{}(p.second);
		return h1 ^ (h2 << 1);
	}
};

// 缺陷检测结果结构体（单轮廓：最大面积外轮廓）
struct DefectObject {
	int class_id;                   // 缺陷类别ID（0=划痕，1=孔洞等）
	std::vector<cv::Point> contour; // 最大面积的外轮廓（单轮廓）
	cv::Rect bbox;                  // 缺陷包围盒
	float score;                    // 置信度
	cv::Point2f centroid;           // 缺陷中心（全局坐标）
	cv::Point global_offset;        // 子图在大图中的偏移
	double contour_area;            // 最大轮廓面积
	double contour_perimeter;       // 最大轮廓周长

	// 构造函数：从掩码生成最大面积外轮廓
	DefectObject(int cid, const cv::Mat& mask, float s, const cv::Point& offset)
		: class_id(cid), score(s), global_offset(offset) {
		// 提取所有外轮廓
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// 选择最大面积的轮廓
		if (!contours.empty()) {
			auto max_contour_it = std::max_element(contours.begin(), contours.end(),
				[](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
					return cv::contourArea(a) < cv::contourArea(b);
				});
			contour = *max_contour_it;

			// 更新轮廓到大图坐标
			for (auto& pt : contour) {
				pt.x += offset.x;
				pt.y += offset.y;
			}
		}

		// 计算包围盒、中心、面积、周长
		bbox = cv::boundingRect(contour);
		centroid = cv::Point2f(bbox.x + bbox.width / 2,
			bbox.y + bbox.height / 2);
		contour_area = cv::contourArea(contour);
		contour_perimeter = cv::arcLength(contour, true);
	}
};

// 空间分桶融合器：处理跨子图缺陷融合
class SpatialBucketDefectMerger {
private:
	cv::Size bucket_size_;                // 分桶大小
	int neighborhood_size_;               // 邻域大小
	float max_distance_threshold_;        // 中心距离阈值
	std::unordered_map<std::pair<int, int>, std::vector<DefectObject>, PairHash> buckets_;  // 全局分桶

	// 计算缺陷中心所属的桶坐标
	std::pair<int, int> getBucketKey(const cv::Point2f& centroid) const {
		int bucket_x = static_cast<int>(centroid.x / bucket_size_.width);
		int bucket_y = static_cast<int>(centroid.y / bucket_size_.height);
		return { bucket_x, bucket_y };
	}

	// 获取邻域桶列表
	std::vector<std::pair<int, int>> getNeighborhoodBuckets(const std::pair<int, int>& bucket_key) const {
		std::vector<std::pair<int, int>> neighborhood;
		int bx = bucket_key.first;
		int by = bucket_key.second;

		for (int dx = -neighborhood_size_; dx <= neighborhood_size_; ++dx) {
			for (int dy = -neighborhood_size_; dy <= neighborhood_size_; ++dy) {
				neighborhood.emplace_back(bx + dx, by + dy);
			}
		}
		return neighborhood;
	}

	// 计算两个单轮廓的IoU
	float calculateContourIoU(const std::vector<cv::Point>& contour1,
		const std::vector<cv::Point>& contour2,
		const cv::Rect& bbox1, const cv::Rect& bbox2) const {
		cv::Rect union_bbox = bbox1 | bbox2;
		if (union_bbox.width <= 0 || union_bbox.height <= 0) return 0.0f;

		cv::Mat mask1 = cv::Mat::zeros(union_bbox.size(), CV_8UC1);
		cv::Mat mask2 = cv::Mat::zeros(union_bbox.size(), CV_8UC1);

		// 平移轮廓到掩码坐标系
		std::vector<cv::Point> contour1_shifted = contour1;
		std::vector<cv::Point> contour2_shifted = contour2;
		for (auto& pt : contour1_shifted) {
			pt.x -= union_bbox.x;
			pt.y -= union_bbox.y;
		}
		for (auto& pt : contour2_shifted) {
			pt.x -= union_bbox.x;
			pt.y -= union_bbox.y;
		}

		// 绘制轮廓掩码
		cv::drawContours(mask1, std::vector<std::vector<cv::Point> >{contour1_shifted}, 0, 255, cv::FILLED);
		cv::drawContours(mask2, std::vector<std::vector<cv::Point> >{ contour2_shifted }, 0, 255, cv::FILLED);

		// 计算IoU
		cv::Mat inter, union_mat;
		cv::bitwise_and(mask1, mask2, inter);
		cv::bitwise_or(mask1, mask2, union_mat);

		float inter_area = cv::countNonZero(inter);
		float union_area = cv::countNonZero(union_mat);

		return union_area > 0 ? inter_area / union_area : 0.0f;
	}

	// 面积相似度
	float calculateAreaSimilarity(double area1, double area2) const {
		double min_area = std::min(area1, area2);
		double max_area = std::max(area1, area2);
		return max_area > 0 ? static_cast<float>(min_area / max_area) : 0.0f;
	}

	// 形状相似度（矩形度+圆度）
	float calculateShapeSimilarity(const DefectObject& defect1, const DefectObject& defect2) const {
		// 矩形度
		float rectness1 = static_cast<float>(defect1.contour_area / (defect1.bbox.width * defect1.bbox.height));
		float rectness2 = static_cast<float>(defect2.contour_area / (defect2.bbox.width * defect2.bbox.height));
		float rectness_sim = 1.0f - std::abs(rectness1 - rectness2);

		// 圆度
		float circularity1 = (defect1.contour_perimeter > 0) ? static_cast<float>(4 * CV_PI * defect1.contour_area / (defect1.contour_perimeter * defect1.contour_perimeter)) : 0.0f;
		float circularity2 = (defect2.contour_perimeter > 0) ? static_cast<float>(4 * CV_PI * defect2.contour_area / (defect2.contour_perimeter * defect2.contour_perimeter)) : 0.0f;
		float circularity_sim = 1.0f - std::abs(circularity1 - circularity2);

		return rectness_sim * circularity_sim;
	}

	// 中心距离
	float calculateCenterDistance(const DefectObject& defect1, const DefectObject& defect2) const {
		return static_cast<float>(cv::norm(defect1.centroid - defect2.centroid));
	}

	// 多维度相似度（含距离约束）
	float calculateContourSimilarity(const DefectObject& defect1, const DefectObject& defect2, bool& distance_pass) const {
		float distance = calculateCenterDistance(defect1, defect2);
		distance_pass = (distance <= max_distance_threshold_);
		if (!distance_pass) return 0.0f;

		float iou = calculateContourIoU(defect1.contour, defect2.contour, defect1.bbox, defect2.bbox);
		float area_sim = calculateAreaSimilarity(defect1.contour_area, defect2.contour_area);
		float shape_sim = calculateShapeSimilarity(defect1, defect2);

		const float w_iou = 0.5f;
		const float w_area = 0.2f;
		const float w_shape = 0.3f;
		return w_iou * iou + w_area * area_sim + w_shape * shape_sim;
	}

	// 合并两个缺陷的单轮廓
	DefectObject mergeDefects(const DefectObject& defect1, const DefectObject& defect2) const {
		std::vector<cv::Point> merged_contour = defect1.contour;
		merged_contour.insert(merged_contour.end(), defect2.contour.begin(), defect2.contour.end());

		cv::Rect merged_bbox = defect1.bbox | defect2.bbox;
		double merged_area = cv::contourArea(merged_contour);
		double merged_perimeter = cv::arcLength(merged_contour, true);

		cv::Mat dummy_mask(merged_bbox.size(), CV_8UC1, cv::Scalar::all(0));
		for (auto& pt : merged_contour) {
			dummy_mask.at<uchar>(pt - merged_bbox.tl()) = 255;
		}
		DefectObject merged_defect(defect1.class_id, dummy_mask, (defect1.score + defect2.score) / 2, merged_bbox.tl());
		merged_defect.contour = merged_contour;
		merged_defect.bbox = merged_bbox;
		merged_defect.centroid = cv::Point2f(merged_bbox.x + merged_bbox.width / 2,
			merged_bbox.y + merged_bbox.height / 2);
		merged_defect.contour_area = merged_area;
		merged_defect.contour_perimeter = merged_perimeter;

		return merged_defect;
	}

public:
	// 构造函数
	SpatialBucketDefectMerger(const cv::Size& bucket_size, int neighborhood_size = 1, float max_distance = 300.0f)
		: bucket_size_(bucket_size), neighborhood_size_(neighborhood_size), max_distance_threshold_(max_distance) {}

	// 添加子图缺陷到分桶
	void addSubgraphDefects(const std::vector<DefectObject>& defects) {
		for (const auto& defect : defects) {
			auto key = getBucketKey(defect.centroid);
			buckets_[key].push_back(defect);
		}
	}

	// 融合跨子图缺陷
	std::vector<DefectObject> mergeCrossSubgraphDefects(float similarity_threshold = 0.5f) {
		std::vector<DefectObject> merged_defects;
		std::unordered_map<const DefectObject*, bool> is_merged;

		for (auto& [bucket_key, defects] : buckets_) {
			for (size_t i = 0; i < defects.size(); ++i) {
				const DefectObject* current_defect = &defects[i];
				if (is_merged.count(current_defect) && is_merged[current_defect]) {
					continue;
				}

				DefectObject merged_candidate = *current_defect;
				bool merged = false;

				auto neighborhood = getNeighborhoodBuckets(bucket_key);
				for (const auto& neighbor_key : neighborhood) {
					if (buckets_.find(neighbor_key) == buckets_.end()) {
						continue;
					}

					const auto& neighbor_defects = buckets_[neighbor_key];
					for (const auto& neighbor_defect : neighbor_defects) {
						if (&neighbor_defect == current_defect || is_merged.count(&neighbor_defect)) {
							continue;
						}
						if (merged_candidate.class_id != neighbor_defect.class_id) {
							continue;
						}

						bool distance_pass;
						float similarity = calculateContourSimilarity(merged_candidate, neighbor_defect, distance_pass);

						if (similarity >= similarity_threshold && distance_pass) {
							merged_candidate = mergeDefects(merged_candidate, neighbor_defect);
							is_merged[&neighbor_defect] = true;
							merged = true;
						}
					}
				}

				if (!merged && !is_merged.count(&merged_candidate)) {
					merged_defects.push_back(merged_candidate);
					is_merged[&defects[i]] = true;
				}
				else if (merged) {
					merged_defects.push_back(merged_candidate);
				}
			}
		}

		return merged_defects;
	}

	// 重置分桶
	void reset() {
		buckets_.clear();
	}
};

// 大图切片函数（640×640，重叠200）
std::vector<std::tuple<cv::Mat, cv::Point>> sliceLargeImage(const cv::Mat& large_img,
	int subgraph_size = 640,
	int overlap = 200) {
	std::vector<std::tuple<cv::Mat, cv::Point>> subgraphs;
	int step = subgraph_size - overlap;

	for (int y = 0; y < large_img.rows; y += step) {
		for (int x = 0; x < large_img.cols; x += step) {
			int x_end = std::min(x + subgraph_size, large_img.cols);
			int y_end = std::min(y + subgraph_size, large_img.rows);
			int w = x_end - x;
			int h = y_end - y;

			cv::Mat subgraph = large_img(cv::Rect(x, y, w, h)).clone();
			subgraphs.emplace_back(subgraph, cv::Point(x, y));

			if (y_end == large_img.rows) break;
		}
		if (y + step >= large_img.rows) break;
	}

	return subgraphs;
}

//合并大图中YOLO-seg的检测结果
int test_merge_yolo_seg_rslts() {
	//TODO::尽管能够跑通，但是结果与预期不符。 
	// 1. 模拟32768×32768大图
	std::cout << "正在创建模拟大图..." << std::endl;
	cv::Mat large_img = cv::Mat::zeros(32768, 32768, CV_8UC1);
	int class_scratch = 0; // 划痕
	int class_hole = 1;    // 孔洞
	// 绘制模拟缺陷
	cv::line(large_img, cv::Point(500, 500), cv::Point(1000, 1000), cv::Scalar(255, 255, 255), 50);
	cv::circle(large_img, cv::Point(2000, 2000), 150, cv::Scalar(125, 0, 0), -1);
	cv::imwrite("./largeImg.png", large_img);
	// 2. 切片大图
	std::cout << "正在切片大图..." << std::endl;
	auto subgraphs = sliceLargeImage(large_img, 640, 200);
	std::cout << "生成子图数量：" << subgraphs.size() << std::endl;

	// 3. 初始化融合器
	SpatialBucketDefectMerger merger(cv::Size(200, 200), 1, 300.0f);
	// 4. 模拟YOLO11-seg多类别检测
	std::cout << "正在模拟缺陷检测..." << std::endl;
	for (const auto& [subgraph, offset] : subgraphs) {
		cv::Mat mask_scratch = cv::Mat::zeros(subgraph.size(), CV_8UC1);
		cv::Mat mask_hole = cv::Mat::zeros(subgraph.size(), CV_8UC1);

		// 模拟划痕掩码
		cv::line(mask_scratch, cv::Point(500 - offset.x, 500 - offset.y),
			cv::Point(1000 - offset.x, 1000 - offset.y), 255, 50);
		// 模拟孔洞掩码
		cv::circle(mask_hole, cv::Point(2000 - offset.x, 2000 - offset.y), 150, 255, -1);

		// 添加缺陷到融合器
		if (cv::countNonZero(mask_scratch) > 0) {
			DefectObject defect_scratch(class_scratch, mask_scratch, 0.95f, offset);
			merger.addSubgraphDefects({ defect_scratch });
		}
		if (cv::countNonZero(mask_hole) > 0) {
			DefectObject defect_hole(class_hole, mask_hole, 0.92f, offset);
			merger.addSubgraphDefects({ defect_hole });
		}
	}

	// 5. 融合跨子图缺陷
	std::cout << "正在融合跨子图缺陷..." << std::endl;
	auto merged_defects = merger.mergeCrossSubgraphDefects(0.5f);
	std::cout << "融合后总缺陷数量：" << merged_defects.size() << std::endl;

	// 统计各类别数量
	int scratch_count = 0, hole_count = 0;
	for (const auto& defect : merged_defects) {
		if (defect.class_id == class_scratch) scratch_count++;
		else if (defect.class_id == class_hole) hole_count++;
	}
	std::cout << "融合后划痕数量：" << scratch_count << std::endl;
	std::cout << "融合后孔洞数量：" << hole_count << std::endl;

	// 6. 绘制结果
	std::cout << "正在保存融合结果..." << std::endl;
	cv::Mat result_img = cv::Mat::zeros(large_img.size(), CV_8UC1);
	for (const auto& defect : merged_defects) {
		cv::Scalar color = (defect.class_id == class_scratch) ? cv::Scalar(255, 255, 255) : cv::Scalar(125, 125, 125);
		cv::drawContours(result_img, std::vector<std::vector<cv::Point>>{ defect.contour }, 0, color, 10);
	}
	cv::imwrite("merged_multi_class_result.png", result_img);

	std::cout << "处理完成！结果已保存为 merged_multi_class_result.png" << std::endl;
	system("pause"); // Windows 暂停控制台
	return 0;
}