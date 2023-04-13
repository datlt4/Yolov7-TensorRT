#include <iostream>
#include "Yolov7TensorRT.h"
#include <chrono>

VizgardLogger::Logger *vizgardLogger = VizgardLogger::LoggerFactory::CreateConsoleLogger(VizgardLogger::INFO);

int main(int argc, char **argv)
{

    Yolov7TRT yolov7;
    yolov7.LoadEngine("/home/kikai/Documents/Yolov7-TensorRT/yolov7-80-class-dla.engine");

    cv::Mat image_bgr = cv::imread("/home/kikai/Documents/Yolov7-TensorRT/yolo-test.jpg");
    std::cout << image_bgr.cols << "  " << image_bgr.rows;
    std::vector<bbox_t> boxes;
    for (int i = 0; i < 100; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now(); // Get current time
        boxes = yolov7.EngineInference(image_bgr);
        auto end = std::chrono::high_resolution_clock::now();            // Get current time
        std::chrono::duration<double, std::milli> elapsed = end - start; // Calculate elapsed time
        std::cout << "Elapsed time: " << elapsed.count() << " ms\n";     // Print elapsed time in milliseconds
    }
    for (bbox_t &b : boxes)
    {
        cv::rectangle(image_bgr, cv::Rect2f(b.x, b.y, b.w, b.h), cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("saved.jpg", image_bgr);
    return 0;
}
