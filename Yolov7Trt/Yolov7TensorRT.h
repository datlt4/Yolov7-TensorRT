#ifndef YOLOV7_TENSORRT_H
#define YOLOV7_TENSORRT_H

#include "../TrtExec/Trtexec.h"
#include "../common/common.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <numeric>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

extern EmoiLogger::Logger* emoiLogger;

class Yolov7TRT : public TrtExec
{
  protected:
    int batch_size = 1;
    std::map<int, std::string> detect_labels{ std::make_pair(0, "Zero"), std::make_pair(1, "One"), std::make_pair(2, "Two"), std::make_pair(3, "Three"), std::make_pair(4, "Four"), std::make_pair(5, "Five"), std::make_pair(6, "Six") };

  public:
    Yolov7TRT();
    ~Yolov7TRT(){};
    bool LoadEngine(const std::string& fileName);
    std::vector<bbox_t> EngineInference(cv::Mat& image);

  private:
    std::vector<float> prepareImage(cv::Mat& img);
    std::vector<bbox_t> postProcess(int src_rows, int src_cols, float* output, size_t outSize);
    bool processInput(float* hostDataBuffer, const int batchSize, cudaStream_t& stream);
    void GenerateReferMatrix();
    float sigmoid(float in);
    void NmsDetect(std::vector<bbox_t>& detections);
    float IOUCalculate(const bbox_t& det_a, const bbox_t& det_b);

    IEmoiLogger iVLogger = IEmoiLogger();

    float obj_threshold = 0.35;
    float nms_threshold = 0.65;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
    std::vector<int> strides = { 8, 16, 32 };
    std::vector<int> num_anchors = { 3, 3, 3 };
    std::vector<std::vector<int>> anchors = { { 12, 16 }, { 19, 36 }, { 40, 28 }, { 36, 75 }, { 76, 55 }, { 72, 146 }, { 142, 110 }, { 192, 243 }, { 459, 401 } };
    std::vector<std::vector<int>> grids;

    const int BATCH_SIZE = 1;
    const int IMAGE_WIDTH = 640;
    const int IMAGE_HEIGHT = 640;
    const int IMAGE_CHANNEL = 3;
    const int CATEGORY = detect_labels.size();
    const int OUTPUT_WIDTH = CATEGORY + 5;

    int x_offset;
    int y_offset;
};

bool readTrtFile(const std::string& engineFile, nvinfer1::ICudaEngine*& engine, IEmoiLogger& iVLogger);

#endif // YOLOV7_TENSORRT_H
