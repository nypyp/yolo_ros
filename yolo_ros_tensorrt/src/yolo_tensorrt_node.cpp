#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "yolo_ros_tensorrt/yolov7.hpp"
#include "yolo_ros_tensorrt/yolov8.hpp"
#include "chrono"
#include "ament_index_cpp/get_package_share_directory.hpp"

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

class YOLOLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public: 
    YOLOLifecycleNode()
    : rclcpp_lifecycle::LifecycleNode("yolo_lifecycle_node"),
        qos_(rclcpp::QoS(10)) {
        declare_parameter<std::string>("model", "yolov7-tiny");
        declare_parameter<std::string>("engine", "yolov7-tiny-nms.trt");
        declare_parameter<int>("input_width", 640);
        declare_parameter<int>("input_height", 640);
        declare_parameter<bool>("enable", true);
        declare_parameter<int>("image_reliability", 2);

        get_parameter("model", model_);
        get_parameter("engine", engine_);
        get_parameter("input_width", input_width_);
        get_parameter("input_height", input_height_);
        get_parameter("enable", enable_);
        get_parameter("image_reliability", image_reliability_);
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn 
    on_configure(const rclcpp_lifecycle::State &state) override {
        RCLCPP_INFO(get_logger(), "[%s] Configuring...", get_name());

        publisher_ = create_publisher<yolo_msgs::msg::DetectionArray>(
            "detections", 
            qos_
        );

        std::string package_share_dir = ament_index_cpp::get_package_share_directory("yolo_ros_tensorrt");
        engine_ = package_share_dir + "/engines/" + engine_;
        
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &state) override {
        RCLCPP_INFO(get_logger(), "[%s] Activating...", get_name());

        cudaSetDevice(0);
        
        if (model_ == "yolov7-tiny") {
            yolo_ = std::make_unique<YOLOv7>(engine_);
        } else if (model_ == "yolov8") {
            yolo_ = std::make_unique<YOLOv8>(engine_);
        } else {
            throw std::runtime_error("Invalid model name");
        }
        yolo_->make_pipe(true);

        rclcpp::ReliabilityPolicy 
            reliability_policy = static_cast<rclcpp::ReliabilityPolicy>(image_reliability_);
        qos_.reliability(reliability_policy);

        subscription_ = create_subscription<sensor_msgs::msg::Image>(
            "image_raw", 
            qos_, 
            std::bind(&YOLOLifecycleNode::image_callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(get_logger(), "[%s] Activated", get_name());
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &state) override {
        RCLCPP_INFO(get_logger(), "[%s] Deactivating...", get_name());

        subscription_.reset();

        RCLCPP_INFO(get_logger(), "[%s] Deactivated", get_name());
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &state) override {
        RCLCPP_INFO(get_logger(), "[%s] Cleaning up...", get_name());

        publisher_.reset();
        yolo_.reset();

        RCLCPP_INFO(get_logger(), "[%s] Cleaned up", get_name());
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }

private: 
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        RCLCPP_DEBUG(get_logger(), "[%s] Image received", get_name());

        if (get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
            cv::Mat cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::Mat res;
            cv::Size size = cv::Size{input_width_, input_height_};
            std::vector<Object> objs;

            yolo_->copy_from_Mat(cv_image, size);

            auto start = std::chrono::steady_clock::now();
            yolo_->infer();
            yolo_->postprocess(objs);
            auto end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            RCLCPP_INFO(get_logger(), "[%s] Inference time: %ld ms", get_name(), duration);
            
            yolo_msgs::msg::DetectionArray detection_array;
            detection_array.header = msg->header;

            for (const auto &obj : objs) {
                yolo_msgs::msg::Detection detection;
                detection.class_id = obj.label;
                detection.class_name = CLASS_NAMES[obj.label];
                detection.score = obj.prob;

                detection.bbox.center.position.x = obj.rect.x + obj.rect.width / 2;
                detection.bbox.center.position.y = obj.rect.y + obj.rect.height / 2;
                detection.bbox.center.theta = 0;

                detection.bbox.size.x = obj.rect.width;
                detection.bbox.size.y = obj.rect.height;

                detection_array.detections.push_back(detection);
            }
            publisher_->publish(detection_array);
        }
    }
    
    std::string model_;
    std::string engine_;
    int input_width_;
    int input_height_;
    bool enable_;
    int image_reliability_;
    rclcpp::QoS qos_;
    std::unique_ptr<YOLO> yolo_;
    rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<YOLOLifecycleNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    node->configure();
    node->activate();
    executor.spin();
    node->deactivate();
    node->cleanup();
    node->shutdown();
    rclcpp::shutdown();
    return 0;
}