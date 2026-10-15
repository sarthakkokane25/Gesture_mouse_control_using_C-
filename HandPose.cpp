#include "HandPose.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
    constexpr int INPUT_WIDTH = 224;
    constexpr int INPUT_HEIGHT = 224;

    constexpr float CONFIDENCE_THRESHOLD = 0.8f;

    // Palm landmark indices from the palm detector.
    constexpr int PALM_BASE = 0;
    constexpr int MIDDLE_FINGER_BASE = 2;

    cv::Point2f transformPoint(
        const cv::Mat& matrix,
        const cv::Point2f& point
    )
    {
        float x =
            static_cast<float>(matrix.at<double>(0, 0)) * point.x +
            static_cast<float>(matrix.at<double>(0, 1)) * point.y +
            static_cast<float>(matrix.at<double>(0, 2));

        float y =
            static_cast<float>(matrix.at<double>(1, 0)) * point.x +
            static_cast<float>(matrix.at<double>(1, 1)) * point.y +
            static_cast<float>(matrix.at<double>(1, 2));

        return cv::Point2f(x, y);
    }
}

HandPose::HandPose(const std::string& modelPath)
    : env(
        ORT_LOGGING_LEVEL_WARNING,
        "GestureAIHandPose"
    ),
      sessionOptions(),
      session(nullptr)
{
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    sessionOptions.SetIntraOpNumThreads(2);

    try
    {
        session = std::make_unique<Ort::Session>(
            env,
            std::wstring(
                modelPath.begin(),
                modelPath.end()
            ).c_str(),
            sessionOptions
        );

        std::cout << "Hand-pose ONNX model loaded successfully.\n";
    }
    catch (const Ort::Exception& e)
    {
        std::cerr
            << "ERROR: Could not load hand-pose model.\n"
            << e.what()
            << std::endl;

        session.reset();
    }
}

std::vector<cv::Point3f> HandPose::detect(
    const cv::Mat& image,
    const std::vector<float>& palm
)
{
    std::vector<cv::Point3f> landmarks;

    if (image.empty())
        return landmarks;

    if (!session)
        return landmarks;

    if (palm.size() < 18)
        return landmarks;

    try
    {
        // =====================================================
        // 1. Read palm bounding box
        // =====================================================

        cv::Point2f palmTopLeft(
            palm[0],
            palm[1]
        );

        cv::Point2f palmBottomRight(
            palm[2],
            palm[3]
        );

        cv::Point2f palmSize =
            palmBottomRight - palmTopLeft;

        if (palmSize.x <= 1.0f || palmSize.y <= 1.0f)
            return landmarks;

        // =====================================================
        // 2. Read palm landmarks
        //
        // palm[4..17] = 7 points × 2
        // =====================================================

        std::vector<cv::Point2f> palmLandmarks;

        for (int i = 0; i < 7; ++i)
        {
            palmLandmarks.emplace_back(
                palm[4 + i * 2],
                palm[4 + i * 2 + 1]
            );
        }

        // =====================================================
        // 3. Enlarge palm region before rotation
        // =====================================================

        cv::Point2f center =
            (palmTopLeft + palmBottomRight) * 0.5f;

        float squareSize =
            std::max(palmSize.x, palmSize.y) * 4.0f;

        cv::Point2f cropTopLeft(
            center.x - squareSize * 0.5f,
            center.y - squareSize * 0.5f
        );

        cv::Point2f cropBottomRight(
            center.x + squareSize * 0.5f,
            center.y + squareSize * 0.5f
        );

        cropTopLeft.x =
            std::max(0.0f, cropTopLeft.x);

        cropTopLeft.y =
            std::max(0.0f, cropTopLeft.y);

        cropBottomRight.x =
            std::min(
                static_cast<float>(image.cols),
                cropBottomRight.x
            );

        cropBottomRight.y =
            std::min(
                static_cast<float>(image.rows),
                cropBottomRight.y
            );

        int cropX =
            static_cast<int>(cropTopLeft.x);

        int cropY =
            static_cast<int>(cropTopLeft.y);

        int cropWidth =
            static_cast<int>(
                cropBottomRight.x - cropTopLeft.x
            );

        int cropHeight =
            static_cast<int>(
                cropBottomRight.y - cropTopLeft.y
            );

        if (cropWidth <= 1 || cropHeight <= 1)
            return landmarks;

        cv::Mat crop =
            image(
                cv::Rect(
                    cropX,
                    cropY,
                    cropWidth,
                    cropHeight
                )
            ).clone();

        // =====================================================
        // 4. Calculate hand rotation
        // =====================================================

        cv::Point2f p1 =
            palmLandmarks[PALM_BASE];

        cv::Point2f p2 =
            palmLandmarks[MIDDLE_FINGER_BASE];

        p1.x -= static_cast<float>(cropX);
        p1.y -= static_cast<float>(cropY);

        p2.x -= static_cast<float>(cropX);
        p2.y -= static_cast<float>(cropY);

        float angleRadians =
            static_cast<float>(
                CV_PI / 2.0 -
                std::atan2(
                    -(p2.y - p1.y),
                    p2.x - p1.x
                )
            );

        float angleDegrees =
            angleRadians * 180.0f /
            static_cast<float>(CV_PI);

        cv::Point2f rotationCenter(
            crop.cols * 0.5f,
            crop.rows * 0.5f
        );

        cv::Mat rotation =
            cv::getRotationMatrix2D(
                rotationCenter,
                angleDegrees,
                1.0
            );

        cv::Mat rotated;

        cv::warpAffine(
            crop,
            rotated,
            rotation,
            crop.size()
        );

        // =====================================================
        // 5. Convert BGR → RGB
        // =====================================================

        cv::cvtColor(
            rotated,
            rotated,
            cv::COLOR_BGR2RGB
        );

        // =====================================================
        // 6. Resize to 224 × 224
        // =====================================================

        cv::Mat resized;

        cv::resize(
            rotated,
            resized,
            cv::Size(
                INPUT_WIDTH,
                INPUT_HEIGHT
            ),
            0,
            0,
            cv::INTER_AREA
        );

        // =====================================================
        // 7. Create NHWC float input
        //
        // [1, 224, 224, 3]
        // =====================================================

        std::vector<float> inputTensor(
            INPUT_WIDTH *
            INPUT_HEIGHT *
            3
        );

        for (int y = 0; y < INPUT_HEIGHT; ++y)
        {
            for (int x = 0; x < INPUT_WIDTH; ++x)
            {
                const cv::Vec3b& pixel =
                    resized.at<cv::Vec3b>(y, x);

                int index =
                    (y * INPUT_WIDTH + x) * 3;

                inputTensor[index + 0] =
                    pixel[0] / 255.0f;

                inputTensor[index + 1] =
                    pixel[1] / 255.0f;

                inputTensor[index + 2] =
                    pixel[2] / 255.0f;
            }
        }

        // =====================================================
        // 8. ONNX input shape
        // =====================================================

        std::array<int64_t, 4> inputShape =
        {
            1,
            INPUT_HEIGHT,
            INPUT_WIDTH,
            3
        };

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        Ort::Value inputValue =
            Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputTensor.data(),
                inputTensor.size(),
                inputShape.data(),
                inputShape.size()
            );

        // =====================================================
        // 9. Get model input/output names
        // =====================================================

        Ort::AllocatorWithDefaultOptions allocator;

        auto inputNameAllocated =
            session->GetInputNameAllocated(
                0,
                allocator
            );

        const char* inputName =
            inputNameAllocated.get();

        size_t outputCount =
            session->GetOutputCount();

        std::vector<
            Ort::AllocatedStringPtr
        > outputNameStorage;

        std::vector<const char*> outputNames;

        outputNameStorage.reserve(outputCount);
        outputNames.reserve(outputCount);

        for (size_t i = 0; i < outputCount; ++i)
        {
            outputNameStorage.push_back(
                session->GetOutputNameAllocated(
                    i,
                    allocator
                )
            );

            outputNames.push_back(
                outputNameStorage.back().get()
            );
        }

        // =====================================================
        // 10. Run ONNX model
        // =====================================================

        const char* inputNames[] =
        {
            inputName
        };

        auto outputValues =
            session->Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputValue,
                1,
                outputNames.data(),
                outputNames.size()
            );

        if (outputValues.size() < 4)
        {
            std::cerr
                << "ERROR: Hand-pose model returned "
                << outputValues.size()
                << " outputs. Expected at least 4."
                << std::endl;

            return landmarks;
        }

        // =====================================================
        // 11. Read confidence
        // =====================================================

        float confidence =
            outputValues[1]
                .GetTensorData<float>()[0];

        if (confidence < CONFIDENCE_THRESHOLD)
            return landmarks;

        // =====================================================
        // 12. Read 21 × 3 screen landmarks
        // =====================================================

        const float* landmarkData =
            outputValues[0]
                .GetTensorData<float>();

        // =====================================================
        // 13. Read rotated palm bounding box
        // =====================================================

        std::vector<cv::Point2f>
            rotatedPalmPoints;

        for (const auto& point : palmLandmarks)
        {
            cv::Point2f localPoint(
                point.x - cropX,
                point.y - cropY
            );

            cv::Point2f rotatedPoint =
                transformPoint(
                    rotation,
                    localPoint
                );

            rotatedPalmPoints.push_back(
                rotatedPoint
            );
        }

        float minX =
            std::numeric_limits<float>::max();

        float minY =
            std::numeric_limits<float>::max();

        float maxX =
            std::numeric_limits<float>::lowest();

        float maxY =
            std::numeric_limits<float>::lowest();

        for (const auto& point :
             rotatedPalmPoints)
        {
            minX = std::min(minX, point.x);
            minY = std::min(minY, point.y);
            maxX = std::max(maxX, point.x);
            maxY = std::max(maxY, point.y);
        }

        float rotatedWidth =
            maxX - minX;

        float rotatedHeight =
            maxY - minY;

        float scale =
            std::max(
                rotatedWidth,
                rotatedHeight
            ) * 3.0f;

        if (scale <= 1.0f)
            return landmarks;

        // =====================================================
        // 14. Convert model coordinates back
        //     to rotated-image coordinates
        // =====================================================

        cv::Point2f rotatedCenter(
            (minX + maxX) * 0.5f,
            (minY + maxY) * 0.5f
        );

        std::vector<cv::Point2f>
            rotatedLandmarks;

        for (int i = 0; i < 21; ++i)
        {
            float modelX =
                landmarkData[i * 3 + 0];

            float modelY =
                landmarkData[i * 3 + 1];

            float modelZ =
                landmarkData[i * 3 + 2];

            // Model coordinates are relative to the
            // center of the 224 × 224 input.
            float x =
                (modelX - INPUT_WIDTH * 0.5f) *
                (scale / INPUT_WIDTH);

            float y =
                (modelY - INPUT_HEIGHT * 0.5f) *
                (scale / INPUT_HEIGHT);

            x += rotatedCenter.x;
            y += rotatedCenter.y;

            rotatedLandmarks.emplace_back(
                x,
                y
            );

            (void)modelZ;
        }

        // =====================================================
        // 15. Invert rotation
        // =====================================================

        cv::Mat inverseRotation;

        cv::invertAffineTransform(
            rotation,
            inverseRotation
        );

        // =====================================================
        // 16. Convert to original camera coordinates
        // =====================================================

        for (int i = 0; i < 21; ++i)
        {
            cv::Point2f originalPoint =
                transformPoint(
                    inverseRotation,
                    rotatedLandmarks[i]
                );

            originalPoint.x += cropX;
            originalPoint.y += cropY;

            originalPoint.x =
                std::clamp(
                    originalPoint.x,
                    0.0f,
                    static_cast<float>(
                        image.cols - 1
                    )
                );

            originalPoint.y =
                std::clamp(
                    originalPoint.y,
                    0.0f,
                    static_cast<float>(
                        image.rows - 1
                    )
                );

            float z =
                landmarkData[i * 3 + 2];

            landmarks.emplace_back(
                originalPoint.x,
                originalPoint.y,
                z
            );
        }

        return landmarks;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr
            << "ONNX Runtime error in HandPose::detect():\n"
            << e.what()
            << std::endl;

        return {};
    }
    catch (const cv::Exception& e)
    {
        std::cerr
            << "OpenCV error in HandPose::detect():\n"
            << e.what()
            << std::endl;

        return {};
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "Standard error in HandPose::detect():\n"
            << e.what()
            << std::endl;

        return {};
    }
}