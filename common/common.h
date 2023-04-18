#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct bbox_t
{
    float x, y, w,
        h;             // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;        // confidence - probability that the object was found correctly
    int obj_id;        // class of object - from range [0, classes-1]
    int track_id = -1; // tracking id for video (0 - untracked, 1 - inf - tracked object)
    bool is_inside_zone = 0;
    std::vector<cv::Point2f> keypoints = {};
    std::vector<float> kp_scores; // proposal_score
    bool is_lone_person = false;
    float bri = 0.0f;
    int time_in_RZ = 0;
    int time_in_PZ = 0;
    bool is_inside_RZ = 0;
    bool is_inside_PZ = 0;
    std::string name;
    std::string distance;
    std::string message = "";

    void update_xywh(float x, float y, float w, float h)
    {
        this->x = x;
        this->y = y;
        this->w = w;
        this->h = h;
    }
};
