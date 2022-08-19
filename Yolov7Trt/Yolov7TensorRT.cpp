#include "Yolov7TensorRT.h"
using namespace VizgardTrt;

Yolov7TRT::Yolov7TRT() : TrtExec()
{
    int index = 0;
    for (const int &stride : strides)
    {
        grids.push_back({num_anchors[index], int(IMAGE_HEIGHT / stride), int(IMAGE_WIDTH / stride)});
        index++;
    }
    refer_rows = 0;
    refer_cols = 6;
    for (const std::vector<int> &grid : grids)
    {
        refer_rows += std::accumulate(grid.begin(), grid.end(), 1, std::multiplies<int>());
    }
    std::cout << TAGLINE << "refer_rows: " << refer_rows << std::endl;
    GenerateReferMatrix();
}

bool Yolov7TRT::LoadEngine(const std::string &fileName)
{
    bool r = this->loadEngine(fileName);
    assert(r);
    return r;
}

std::vector<bbox_t> Yolov7TRT::EngineInference(cv::Mat &image)
{
    std::vector<bbox_t> boxes;
    std::vector<float> curInput = prepareImage(image);
    if (!curInput.data())
    {
        return boxes;
    }

    this->processInput(curInput.data(), BATCH_SIZE, stream);
    std::vector<void *> predicitonBindings = {(float *)input_buffers[0], (float *)output_buffers[0]};
    // VLOG(INFO) << "Input " << log_cuda_bf(prediction_input_dims[0], predicitonBindings[0], 100);
    this->prediction_context->setBindingDimensions(0, nvinfer1::Dims4(BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH));
    this->prediction_context->enqueue(BATCH_SIZE, predicitonBindings.data(), 0, nullptr);
    // VLOG(INFO) << "Output: " << log_cuda_bf(prediction_output_dims[0], predicitonBindings[1], 200);
    std::vector<float> output(BATCH_SIZE * refer_rows * OUTPUT_WIDTH);
    cudaMemcpy(output.data(), predicitonBindings[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    boxes = postProcess(image.rows, image.cols, output.data(), output.size());
    std::sort(boxes.begin(), boxes.end(),
              [](bbox_t A, bbox_t B)
              {
                  // return (A.prob > B.prob);
                  return ((static_cast<float>(A.w) * static_cast<float>(A.h)) > (static_cast<float>(B.h) * static_cast<float>(B.w)));
              });
    return boxes;
}

std::vector<float> Yolov7TRT::prepareImage(cv::Mat &img)
{
    std::vector<float> result(long(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL));
    float *data = result.data();
    int index = 0;
    if (!img.data)
        return result;

    cv::Mat flt_img;
    float ratio = std::min(float(IMAGE_WIDTH) / float(img.cols), float(IMAGE_HEIGHT) / float(img.rows));
    flt_img = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32FC3, 0.5);
    cv::Mat rsz_img;
    cv::resize(img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.convertTo(rsz_img, CV_32FC3, 1.0 / 255);
    // x_offset = (IMAGE_WIDTH - rsz_img.cols) / 2;
    // y_offset = (IMAGE_HEIGHT - rsz_img.rows) / 2;
    x_offset = 0;
    y_offset = 0;
    rsz_img.copyTo(flt_img(cv::Rect(x_offset, y_offset, rsz_img.cols, rsz_img.rows)));

    // HWC TO CHW
    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 2),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data)};
    cv::split(flt_img, split_img);
    return result;
}

bool Yolov7TRT::processInput(float *hostDataBuffer, const int batchSize, cudaStream_t &stream)
{
    // std::vector< void* > input_buffers(this->prediction_engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < this->prediction_engine->getNbBindings(); ++i)
    {
        int32_t binding_size = volume(this->prediction_engine->getBindingDimensions(i)) * batchSize * sizeof(float);
        binding_size = (binding_size > 0) ? binding_size : -binding_size;
        // std::cout << "Size of: " << binding_size << std::endl;
        if (this->prediction_engine->bindingIsInput(i))
        {
            input_buffers.emplace_back(new float());
            cudaMalloc(&input_buffers.back(), binding_size);
            prediction_input_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
        else
        {
            output_buffers.emplace_back(new float());
            cudaMalloc(&output_buffers.back(), binding_size);
            prediction_output_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
    }

    if (prediction_input_dims.empty() || prediction_output_dims.empty())
    {
        VLOG(ERROR) << "Expect at least one input and one output for network";
        return false;
    }

    float *gpu_input_0 = (float *)input_buffers[0];

    // TensorRT copy way
    // Host memory for input buffer
    if (cudaMemcpyAsync(gpu_input_0, hostDataBuffer, size_t(batchSize * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL * sizeof(float)), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        VLOG(ERROR) << "Input corrupted or CUDA error, abort ";
        return false;
    }

    return true;
}

void Yolov7TRT::GenerateReferMatrix()
{
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    int position = 0;
    for (int n = 0; n < (int)grids.size(); n++)
    {
        for (int c = 0; c < grids[n][0]; c++)
        {
            std::vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; h++)
            {
                for (int w = 0; w < grids[n][2]; w++)
                {
                    float *row = refer_matrix.ptr<float>(position);
                    row[0] = w;
                    row[1] = grids[n][2];
                    row[2] = h;
                    row[3] = grids[n][1];
                    row[4] = anchor[0];
                    row[5] = anchor[1];
                    position++;
                }
            }
        }
    }
}

std::vector<bbox_t> Yolov7TRT::postProcess(int src_rows, int src_cols, float *output, size_t outSize)
{
    std::vector<bbox_t> result;
    float ratio = float(src_cols) / float(IMAGE_WIDTH) > float(src_rows) / float(IMAGE_HEIGHT) ? float(src_cols) / float(IMAGE_WIDTH) : float(src_rows) / float(IMAGE_HEIGHT);
    float *out = output;
    cv::Mat result_matrix = cv::Mat(refer_rows, OUTPUT_WIDTH, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++)
    {
        bbox_t box;
        float *row = result_matrix.ptr<float>(row_num);

        if (row[4] < obj_threshold)
            continue;
        auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
        box.prob = row[4] * row[max_pos - row];
        box.obj_id = static_cast<unsigned int>(max_pos - row - 5);

        box.x = (row[0] - row[2] / 2.0) * ratio;
        box.y = (row[1] - row[3] / 2.0) * ratio;
        box.w = row[2] * ratio;
        box.h = row[3] * ratio;

        result.push_back(box);
    }
    NmsDetect(result);
    return result;
}

float Yolov7TRT::sigmoid(float in)
{
    if (1)
        return in;
    else
        return 1.f / (1.f + exp(-in));
}

void Yolov7TRT::NmsDetect(std::vector<bbox_t> &detections)
{
    sort(detections.begin(), detections.end(), [=](const bbox_t &left, const bbox_t &right)
         { return left.prob > right.prob; });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].obj_id == detections[j].obj_id)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const bbox_t &det)
                                    { return det.prob == 0; }),
                     detections.end());
}

float Yolov7TRT::IOUCalculate(const bbox_t &det_a, const bbox_t &det_b)
{
    cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
    cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
    cv::Point2f left_up(std::min(det_a.x, det_b.x), std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w), std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else if (1)
        return inter_area / union_area - distance_d / distance_c;
    else
        return inter_area / union_area - distance_d / distance_c;
}
