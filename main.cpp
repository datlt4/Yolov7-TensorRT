#include <iostream>
#include "Yolov7TensorRT.h"

VizgardLogger::Logger *vizgardLogger = VizgardLogger::LoggerFactory::CreateConsoleLogger(VizgardLogger::INFO);

int main(int argc, char **argv)
{

    Yolov7TRT yolov7;
    yolov7.LoadEngine("./yolov7.engine");

    cv::Mat image_bgr = cv::imread("./person.jpg");
    std::vector<bbox_t> boxes = yolov7.EngineInference(image_bgr);
    std::cout << TAGLINE << boxes.size() << std::endl;
    for (bbox_t &b : boxes)
    {
        cv::rectangle(image_bgr, cv::Rect2f(b.x, b.y, b.w, b.h), cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("saved.jpg", image_bgr);
    return 0;
}