//
// Created by ubuntu on 12/26/24.
//
#ifndef YOLO_TENSORRT_YOLOV7_HPP
#define YOLO_TENSORRT_YOLOV7_HPP
#include "yolo.hpp"
using namespace det;

class YOLOv7 : public YOLO {
public:
    explicit YOLOv7(const std::string& engine_file_path);
    
    void postprocess(std::vector<Object>& objs);
};

YOLOv7::YOLOv7(const std::string& engine_file_path) : YOLO(engine_file_path) {};

void YOLOv7::postprocess(std::vector<Object>& objs)
{
    objs.clear();

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat num_dets = cv::Mat(1, 1, CV_32S, static_cast<int*>(this->host_ptrs[0]));
    cv::Mat det_boxes = cv::Mat(100, 4, CV_32F, static_cast<float*>(this->host_ptrs[1]));
    cv::Mat det_scores = cv::Mat(1, 100, CV_32F, static_cast<float*>(this->host_ptrs[2]));
    cv::Mat det_classes = cv::Mat(1, 100, CV_32S, static_cast<int*>(this->host_ptrs[3]));
    int num_dets_value = num_dets.at<int>(0,0);
    for (int i = 0; i < num_dets_value; i++) {
        // 获取检测框的相关信息
        float score = det_scores.at<float>(0, i);
        int label = static_cast<int>(det_classes.at<int>(0, i));
        float x0 = det_boxes.at<float>(i, 0);
        float y0 = det_boxes.at<float>(i, 1);
        float x1 = det_boxes.at<float>(i, 2);
        float y1 = det_boxes.at<float>(i, 3);
        // 将坐标从 letterbox 尺度映射回原图像尺度
        x0 = (x0 - dw) * ratio;
        y0 = (y0 - dh) * ratio;
        x1 = (x1 - dw) * ratio;
        y1 = (y1 - dh) * ratio;

        // 将处理后的检测框信息保存
        cv::Rect_<float> new_bbox;
        new_bbox.x = x0;
        new_bbox.y = y0;
        new_bbox.width = x1 - x0;
        new_bbox.height = y1 - y0;
        bboxes.push_back(new_bbox);
        labels.push_back(label);
        scores.push_back(score);
        Object obj;
        obj.rect = new_bbox;
        obj.prob  = score;
        obj.label = label;
        objs.push_back(obj);
    }
}
#endif //YOLO_TENSORRT_YOLOV7_HPP
