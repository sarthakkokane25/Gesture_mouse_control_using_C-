#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class MPPalmDet;

class PalmDetector
{
public:
    PalmDetector(
        const std::string& modelPath,
        float scoreThreshold = 0.5f,
        float nmsThreshold = 0.3f
    );

    ~PalmDetector();

    std::vector<std::vector<float>> detect(
        const cv::Mat& image
    );

private:
    MPPalmDet* detector;
};