#ifndef CALIBRATOR_H
#define CALIBRATOR_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#ifndef TAGLINE
#define TAGLINE "\t< " << __FUNCTION__ << " - " << __FILE__ << ":" << __LINE__ << " > "
#endif // TAGLINE

// Define a custom data loader for your data
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
  public:
    Int8Calibrator(std::string filenamePattern, int batchSize, int inputC, int inputH, int inputW)
        : mBatchSize(batchSize)
        , mInputC(inputC)
        , mInputH(inputH)
        , mInputW(inputW)
    {
        std::cout << TAGLINE << std::endl;
        if (glob(filenamePattern.c_str(), 0, NULL, &glob_result) != 0) {
            // handle error
        }
        mDeviceInput = allocateGPUMemory(mBatchSize * mInputH * mInputW * mInputC * sizeof(float));
        mCurBatchData = new float[mBatchSize * mInputH * mInputW * mInputC];
    }

    virtual ~Int8Calibrator()
    {
        std::cout << TAGLINE << std::endl;
        if (mDeviceInput) {
            cudaFree(mDeviceInput);
            mDeviceInput = nullptr;
        }
        globfree(&glob_result);
        delete[] mCurBatchData;
    }

    int getBatchSize() const noexcept override
    {
        std::cout << TAGLINE << " - mBatchSize: " << mBatchSize << std::endl;
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        std::cout << "nbBindings: " << nbBindings << std::endl;
        std::cout << TAGLINE << std::endl;
        std::cout << "mCurBatch: " << mCurBatch << " + mBatchSize: " << mBatchSize << " = " << mCurBatch + mBatchSize << ", gl_pathc: " << glob_result.gl_pathc << std::endl;
        if (mCurBatch + mBatchSize > glob_result.gl_pathc) {
            return false;
        }
        std::cout << TAGLINE << std::endl;
        for (int i = 0; i < mBatchSize; i++) {
            const std::string inputFileName = glob_result.gl_pathv[mCurBatch++];
            readImageData(inputFileName, &mCurBatchData[i * mInputH * mInputW * mInputC]);
        }
        std::cout << TAGLINE << std::endl;

        // Set the input bindings for the network
        for (int i = 0; i < nbBindings; i++) {
            const int bindingIdx = i;
            const char* bindingName = names[bindingIdx];
            void* bindingData = bindings[bindingIdx];

            if (std::string(bindingName) == "images") {
                cudaMemcpy(mDeviceInput, (void*)mCurBatchData, mBatchSize * mInputH * mInputW * mInputC * sizeof(float), cudaMemcpyHostToDevice);
                bindingData = mDeviceInput;
            } else {
                // Handle other input bindings if any
            }
        }
        // cudaMemcpy(bindings[0], reinterpret_cast<void*>(mCurBatchData), mBatchSize * mInputH * mInputW * mInputC * sizeof(float), cudaMemcpyHostToDevice);
        std::cout << TAGLINE << std::endl;
        return true;
    }

    bool getBatch(float* mCurBatchData)
    {
        if (mCurBatch + mBatchSize > glob_result.gl_pathc) {
            return false;
        }
        for (int i = 0; i < mBatchSize; i++) {
            const std::string inputFileName = glob_result.gl_pathv[mCurBatch++];
            std::cout << i << " : " << inputFileName << std::endl;
            readImageData(inputFileName, &mCurBatchData[i * mInputH * mInputW * mInputC]);
        }
        return true;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override
    {
        std::cout << TAGLINE << std::endl;
        std::ifstream cacheFile("calibration.cache", std::ios::binary);
        if (cacheFile.good()) {
            // Get the length of the cache file
            cacheFile.seekg(0, cacheFile.end);
            length = cacheFile.tellg();
            cacheFile.seekg(0, cacheFile.beg);

            // Allocate memory for the cache data
            void* cacheData = malloc(length);
            if (!cacheData) {
                std::cerr << "Failed to allocate memory for calibration cache." << std::endl;
                return nullptr;
            }

            // Read the cache data from the file
            cacheFile.read(reinterpret_cast<char*>(cacheData), length);
            cacheFile.close();

            std::cout << "Read calibration cache successfully. Cache size: " << length << " bytes." << std::endl;
            return cacheData;
        }

        std::cout << "No calibration cache found." << std::endl;
        return nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override
    {
        std::cout << TAGLINE << std::endl;
        std::ofstream cacheFile("calibration.cache", std::ios::binary | std::ios::app);
        if (cacheFile.good()) {
            // Write the cache data to the file
            cacheFile.write(reinterpret_cast<const char*>(ptr), length);
            cacheFile.close();
            std::cout << "Wrote calibration cache successfully. Cache size: " << length << " bytes." << std::endl;
        } else {
            std::cerr << "Failed to open calibration cache file for writing." << std::endl;
        }
    }


  private:
    glob_t glob_result;
    int mBatchSize{ 0 };
    int mInputH{ 0 };
    int mInputW{ 0 };
    int mInputC{ 0 };
    int mCurBatch{ 0 };
    float* mCurBatchData{ nullptr };
    void* mDeviceInput;

  public:
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

    void* allocateGPUMemory(size_t size)
    {
        void* deviceMemory;
        cudaError_t status = cudaMalloc(&deviceMemory, size);
        if (status != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory" << std::endl;
            return nullptr;
        }
        return deviceMemory;
    }
};
#endif // CALIBRATOR_H
