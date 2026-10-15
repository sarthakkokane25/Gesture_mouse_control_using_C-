#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

class HandPose
{
public:
    explicit HandPose(const std::string& modelPath);

    std::vector<cv::Point3f> detect(
        const cv::Mat& image,
        const std::vector<float>& palm
    );

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
};