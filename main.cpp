#include "Yolov7TensorRT.h"
#include <chrono>
#include <iostream>
#include <unordered_map>

EmoiLogger::Logger* emoiLogger = EmoiLogger::LoggerFactory::CreateConsoleLogger(EmoiLogger::INFO);

int main(int argc, char** argv)
{
    // cv::RNG rng(1405); // initialize RNG with seed 0
    // std::vector<cv::Scalar> scalars;

    // for (int i = 0; i < 80; ++i) {
    //     cv::Scalar scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //     scalars.push_back(scalar);
    // }

    // Yolov7TRT yolov7;
    // yolov7.LoadEngine("../yolov7-80-class-fp16.engine");

    // cv::Mat image_bgr = cv::imread("../yolo-test.jpg");
    // std::cout << image_bgr.cols << "  " << image_bgr.rows;
    // std::vector<bbox_t> boxes;
    // for (int i = 0; i < 100; ++i) {
    //     auto start = std::chrono::high_resolution_clock::now();          // Get current time
    //     boxes = yolov7.EngineInference(image_bgr);
    //     auto end = std::chrono::high_resolution_clock::now();            // Get current time
    //     std::chrono::duration<double, std::milli> elapsed = end - start; // Calculate elapsed time
    //     std::cout << "Elapsed time: " << i << " - " << elapsed.count() << " ms\n";     // Print elapsed time in milliseconds
    // }
    // for (bbox_t& b : boxes) {
    //     cv::rectangle(image_bgr, cv::Rect2f(b.x, b.y, b.w, b.h), scalars[b.obj_id], 2);
    // }
    // cv::imwrite("saved.jpg", image_bgr);

    std::string dataForCalibration = "/home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/val2017/*.jpg";
    int mBatchSize = 1;
    int mInputC = 3;
    int mInputH = 640;
    int mInputW = 640;

    std::unique_ptr<Int8Calibrator> calibrator = std::make_unique<Int8Calibrator>(dataForCalibration, mBatchSize, mInputC, mInputH, mInputW);
    float* mCurBatchData = new float[mBatchSize * mInputH * mInputW * mInputC];

    while (calibrator->getBatch(mCurBatchData)) {
        calibrator->writeCalibrationCache((void*)mCurBatchData, mBatchSize * mInputH * mInputW * mInputC * sizeof(float));
    }

    return 0;
}
