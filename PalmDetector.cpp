#include "PalmDetector.h"

#define main mp_palmdet_demo_main
#include "mp_palmdet.cpp"
#undef main

PalmDetector::PalmDetector(
    const std::string& modelPath,
    float scoreThreshold,
    float nmsThreshold
)
    : detector(nullptr)
{
    // MPPalmDet constructor:
    // modelPath, nmsThreshold, scoreThreshold
    detector = new MPPalmDet(
        modelPath,
        nmsThreshold,
        scoreThreshold
    );
}

PalmDetector::~PalmDetector()
{
    delete detector;
    detector = nullptr;
}

std::vector<std::vector<float>> PalmDetector::detect(
    const cv::Mat& image
)
{
    if (detector == nullptr)
        return {};

    if (image.empty())
        return {};

    return detector->infer(image);
}