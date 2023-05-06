#ifndef CALIBRATOR_H
#define CALIBRATOR_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <glob.h>

// Define a custom data loader for your data
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
  public:
    Int8Calibrator(std::string filenamePattern, int batchSize, int inputH, int inputW, int inputC)
        : mBatchSize(batchSize)
        , mInputH(inputH)
        , mInputW(inputW)
        , mInputC(inputC)
    {
        if (glob(filenamePattern.c_str(), 0, NULL, &glob_result) != 0)
        {
            // handle error
        }
        mCurBatchData = new float[mBatchSize * mInputH * mInputW * mInputC];
    }

    virtual ~Int8Calibrator() {
        globfree(&glob_result);
        delete[] mCurBatchData; 
    }

    int getBatchSize() const noexcept override { return mBatchSize; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        if (mCurBatch + mBatchSize > glob_result.gl_pathc) {
            return false;
        }

        for (int i = 0; i < mBatchSize; i++) {
            const std::string inputFileName = glob_result.gl_pathv[mCurBatch++];
            readImageData(inputFileName, &mCurBatchData[i * mInputH * mInputW * mInputC]);
        }

        cudaMemcpy(bindings[0], mCurBatchData, mBatchSize * mInputH * mInputW * mInputC * sizeof(float), cudaMemcpyHostToDevice);

        return true;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override
    {
        // // Load the calibration cache from disk (if it exists)
        // std::ifstream cache("./calibration.cache", std::ios::binary);
        // if (cache.good()) {
        //     std::stringstream buffer;
        //     buffer << cache.rdbuf();
        //     mCalibrationCache = buffer.str();
        // }
        // length = mCalibrationCache.size();
        // return mCalibrationCache.data();
        length = 0;
        return nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override
    {
        // // Save the calibration cache to disk
        // std::ofstream file("./calibration.cache", std::ios::binary);
        // file.write(reinterpret_cast<const char*>(cache), length);
    }

  private:
    glob_t glob_result;
    int mBatchSize{ 0 };
    int mInputH{ 0 };
    int mInputW{ 0 };
    int mInputC{ 0 };
    int mCurBatch{ 0 };
    float* mCurBatchData{ nullptr };

    void readImageData(const std::string& fileName, float* data)
    {
        // TODO: Implement image reading and preprocessing here
        cv::Mat img = cv::imread(fileName);
        if (!img.data)
            return;

        cv::Mat flt_img;
        float ratio = std::min(float(mInputW) / float(img.cols), float(mInputH) / float(img.rows));
        flt_img = cv::Mat(cv::Size(mInputW, mInputH), CV_32FC3, 0.5);
        cv::Mat rsz_img;
        cv::resize(img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.convertTo(rsz_img, CV_32FC3, 1.0 / 255);
        // x_offset = (mInputW - rsz_img.cols) / 2;
        // y_offset = (mInputH - rsz_img.rows) / 2;
        int x_offset = 0;
        int y_offset = 0;
        rsz_img.copyTo(flt_img(cv::Rect(x_offset, y_offset, rsz_img.cols, rsz_img.rows)));

        // HWC TO CHW
        int channelLength = mInputW * mInputH;
        std::vector<cv::Mat> split_img = { cv::Mat(mInputH, mInputW, CV_32FC1, data + channelLength * 2), cv::Mat(mInputH, mInputW, CV_32FC1, data + channelLength), cv::Mat(mInputH, mInputW, CV_32FC1, data) };
        cv::split(flt_img, split_img);
    }
};

#endif // CALIBRATOR_H