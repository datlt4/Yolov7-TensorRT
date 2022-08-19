#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct bbox_t {
	float x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;                    // confidence - probability that the object was found correctly
	int obj_id;           // class of object - from range [0, classes-1]
	int track_id = -1;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    bool is_inside_zone = 0;
    std::vector<cv::Point2f> keypoints = {};
    std::vector<float> kp_scores;   // proposal_score
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

	bool one_hand_in_RZ(std::vector<bool>& kp_in_RZ) {
        if ((kp_in_RZ.at(7) && kp_in_RZ.at(9)) || (kp_in_RZ.at(8) && kp_in_RZ.at(10)))
            return true;
        else return false;
    }

    bool two_hands_in_RZ(std::vector<bool>& kp_in_RZ) {
        if (kp_in_RZ.at(7) || kp_in_RZ.at(9) || kp_in_RZ.at(8) || kp_in_RZ.at(10))
            return true;
        else return false;
    }

    bool a_foot_in_RZ(std::vector<bool>& kp_in_RZ) {
        if (kp_in_RZ.at(15) && kp_in_RZ.at(16))
            return true;
        else return false;
    }

    bool a_knee_in_RZ(std::vector<bool>& kp_in_RZ) {
        if (kp_in_RZ.at(13) && kp_in_RZ.at(14))
            return true;
        else return false;
    }

    bool a_face_in_RZ(std::vector<bool>& kp_in_RZ) {
        if (kp_in_RZ.at(0) && kp_in_RZ.at(1) && kp_in_RZ.at(2) && kp_in_RZ.at(3) && kp_in_RZ.at(4))
            return true;
        else return false;
    }

    bool a_shoulder_in_RZ(std::vector<bool>& kp_in_RZ) {
        if (kp_in_RZ.at(5) || kp_in_RZ.at(6))
            return true;
        else return false;
    }
};

