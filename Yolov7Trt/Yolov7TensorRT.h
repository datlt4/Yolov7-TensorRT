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
    std::map<int, std::string> detect_labels{std::make_pair(1, "person"),
                                             std::make_pair(2, "bicycle"),
                                             std::make_pair(3, "car"),
                                             std::make_pair(4, "motorcycle"),
                                             std::make_pair(5, "airplane"),
                                             std::make_pair(6, "bus"),
                                             std::make_pair(7, "train"),
                                             std::make_pair(8, "truck"),
                                             std::make_pair(9, "boat"),
                                             std::make_pair(10, "traffic light"),
                                             std::make_pair(11, "fire hydrant"),
                                             std::make_pair(12, "stop sign"),
                                             std::make_pair(13, "parking meter"),
                                             std::make_pair(14, "bench"),
                                             std::make_pair(15, "bird"),
                                             std::make_pair(16, "cat"),
                                             std::make_pair(17, "dog"),
                                             std::make_pair(18, "horse"),
                                             std::make_pair(19, "sheep"),
                                             std::make_pair(20, "cow"),
                                             std::make_pair(21, "elephant"),
                                             std::make_pair(22, "bear"),
                                             std::make_pair(23, "zebra"),
                                             std::make_pair(24, "giraffe"),
                                             std::make_pair(25, "backpack"),
                                             std::make_pair(26, "umbrella"),
                                             std::make_pair(27, "handbag"),
                                             std::make_pair(28, "tie"),
                                             std::make_pair(29, "suitcase"),
                                             std::make_pair(30, "Frisbee"),
                                             std::make_pair(31, "skis"),
                                             std::make_pair(32, "snowboard"),
                                             std::make_pair(33, "sports ball"),
                                             std::make_pair(34, "kite"),
                                             std::make_pair(35, "baseball bat"),
                                             std::make_pair(36, "baseball glove"),
                                             std::make_pair(37, "skateboard"),
                                             std::make_pair(38, "surfboard"),
                                             std::make_pair(39, "tennis racket"),
                                             std::make_pair(40, "bottle"),
                                             std::make_pair(41, "wine glass"),
                                             std::make_pair(42, "cup"),
                                             std::make_pair(43, "fork"),
                                             std::make_pair(44, "knife"),
                                             std::make_pair(45, "spoon"),
                                             std::make_pair(46, "bowl"),
                                             std::make_pair(47, "banana"),
                                             std::make_pair(48, "apple"),
                                             std::make_pair(49, "sandwich"),
                                             std::make_pair(50, "orange"),
                                             std::make_pair(51, "broccoli"),
                                             std::make_pair(52, "carrot"),
                                             std::make_pair(53, "hot dog"),
                                             std::make_pair(54, "pizza"),
                                             std::make_pair(55, "donut"),
                                             std::make_pair(56, "cake"),
                                             std::make_pair(57, "chair"),
                                             std::make_pair(58, "couch"),
                                             std::make_pair(59, "potted plant"),
                                             std::make_pair(60, "bed"),
                                             std::make_pair(61, "dining table"),
                                             std::make_pair(62, "toilet"),
                                             std::make_pair(63, "TV"),
                                             std::make_pair(64, "laptop"),
                                             std::make_pair(65, "mouse"),
                                             std::make_pair(66, "remote"),
                                             std::make_pair(67, "keyboard"),
                                             std::make_pair(68, "cell phone"),
                                             std::make_pair(69, "microwave"),
                                             std::make_pair(70, "oven"),
                                             std::make_pair(71, "toaster"),
                                             std::make_pair(72, "sink"),
                                             std::make_pair(73, "refrigerator"),
                                             std::make_pair(74, "book"),
                                             std::make_pair(75, "clock"),
                                             std::make_pair(76, "vase"),
                                             std::make_pair(77, "scissors"),
                                             std::make_pair(78, "teddy bear"),
                                             std::make_pair(79, "hair drier"),
                                             std::make_pair(80, "toothbrush")};

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
