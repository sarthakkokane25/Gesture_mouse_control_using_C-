#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <windows.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>

#include "PalmDetector.h"
#include "HandPose.h"

// ============================================================
// LEFT CLICK
// ============================================================

void leftClick()
{
    INPUT inputs[2] = {};

    inputs[0].type = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

    inputs[1].type = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;

    UINT result = SendInput(
        2,
        inputs,
        sizeof(INPUT)
    );

    if (result != 2)
    {
        std::cerr
            << "Warning: Mouse click could not be sent.\n";
    }
}

// ============================================================
// CHECK LANDMARK VALIDITY
// ============================================================

bool validLandmark(
    const cv::Point3f& point,
    int width,
    int height
)
{
    if (!std::isfinite(point.x) ||
        !std::isfinite(point.y) ||
        !std::isfinite(point.z))
    {
        return false;
    }

    const float margin = 100.0f;

    if (point.x < -margin ||
        point.x > width + margin ||
        point.y < -margin ||
        point.y > height + margin)
    {
        return false;
    }

    return true;
}

// ============================================================
// CLAMP LANDMARK TO CAMERA FRAME
// ============================================================

cv::Point2f clampPoint(
    const cv::Point3f& point,
    int width,
    int height
)
{
    float x = std::clamp(
        point.x,
        0.0f,
        static_cast<float>(width - 1)
    );

    float y = std::clamp(
        point.y,
        0.0f,
        static_cast<float>(height - 1)
    );

    return cv::Point2f(x, y);
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    std::cout
        << "============================================\n";

    std::cout
        << "       Gesture AI Mouse Controller\n";

    std::cout
        << "============================================\n";

    std::cout
        << "Fist -> Open Hand = Exit\n";

    std::cout
        << "Pinch = Left Click\n\n";

    // ========================================================
    // MODEL PATHS
    // ========================================================

    const std::string palmModelPath =
        "Models/"
        "palm_detection_mediapipe_2023feb.onnx";

    const std::string handPoseModelPath =
        "Models/"
        "handpose_estimation_mediapipe_2023feb.onnx";

    // ========================================================
    // CAMERA
    // ========================================================

    cv::VideoCapture camera;

    try
    {
        camera.open(0);
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Camera OpenCV exception: "
            << error.what()
            << "\n";

        return -1;
    }

    if (!camera.isOpened())
    {
        std::cerr
            << "ERROR: Cannot open camera.\n";

        std::cerr
            << "Check that the camera is connected "
            "and not being used by another application.\n";

        return -1;
    }

    camera.set(
        cv::CAP_PROP_FRAME_WIDTH,
        1280
    );

    camera.set(
        cv::CAP_PROP_FRAME_HEIGHT,
        720
    );

    // ========================================================
    // LOAD AI MODELS
    // ========================================================

    PalmDetector* palmDetector = nullptr;
    HandPose* handPose = nullptr;

    try
    {
        palmDetector =
            new PalmDetector(
                palmModelPath
            );

        handPose =
            new HandPose(
                handPoseModelPath
            );
    }
    catch (const Ort::Exception& error)
    {
        std::cerr
            << "ONNX Runtime error:\n"
            << error.what()
            << "\n";

        delete palmDetector;
        delete handPose;

        camera.release();

        return -1;
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "OpenCV model error:\n"
            << error.what()
            << "\n";

        delete palmDetector;
        delete handPose;

        camera.release();

        return -1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Model initialization error:\n"
            << error.what()
            << "\n";

        delete palmDetector;
        delete handPose;

        camera.release();

        return -1;
    }
    catch (...)
    {
        std::cerr
            << "Unknown error while loading models.\n";

        delete palmDetector;
        delete handPose;

        camera.release();

        return -1;
    }

    // ========================================================
    // WINDOW
    // ========================================================

    const std::string windowName =
        "Gesture AI Mouse";

    try
    {
        cv::namedWindow(
            windowName,
            cv::WINDOW_NORMAL
        );

        cv::resizeWindow(
            windowName,
            640,
            360
        );
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Window error: "
            << error.what()
            << "\n";

        delete palmDetector;
        delete handPose;

        camera.release();

        return -1;
    }

    // ========================================================
    // SCREEN SIZE
    // ========================================================

    const int screenWidth =
        GetSystemMetrics(SM_CXSCREEN);

    const int screenHeight =
        GetSystemMetrics(SM_CYSCREEN);

    if (screenWidth <= 0 ||
        screenHeight <= 0)
    {
        std::cerr
            << "ERROR: Could not determine screen size.\n";

        delete palmDetector;
        delete handPose;

        camera.release();
        cv::destroyAllWindows();

        return -1;
    }

    // ========================================================
    // CURSOR POSITION
    // ========================================================

    double smoothX =
        screenWidth / 2.0;

    double smoothY =
        screenHeight / 2.0;

    // ========================================================
    // IMPROVED POINTING SETTINGS
    // ========================================================

    // Normal smoothing
    const double normalSmoothing =
        0.18;

    // Precision smoothing
    const double precisionSmoothing =
        0.08;

    // Ignore very small cursor movements.
    // Increase this if cursor still shakes.
    const double deadZone =
        7.0;

    // Don't send extremely tiny cursor updates.
    const double minimumMove =
        2.0;

    // ========================================================
    // PINCH SETTINGS
    // ========================================================

    const float pinchStart =
        0.62f;

    const float pinchRelease =
        0.90f;

    const int startRequired =
        3;

    const int releaseRequired =
        5;

    // ========================================================
    // PINCH STATE
    // ========================================================

    bool precisionMode =
        false;

    bool clickArmed =
        false;

    int pinchStartFrames =
        0;

    int pinchReleaseFrames =
        0;

    // ========================================================
    // HAND GESTURE EXIT STATE
    // ========================================================

    // Exit sequence:
    // Close hand (fist) -> Open hand
    bool exitFistDetected =
        false;

    int fistFrames =
        0;

    int openFrames =
        0;

    const int fistRequired =
        12;

    const int openRequired =
        12;

    // ========================================================
    // LAST CURSOR POSITION
    // ========================================================

    double lastCursorX =
        smoothX;

    double lastCursorY =
        smoothY;

    // ========================================================
    // MAIN LOOP
    // ========================================================

    bool running =
        true;

    while (running)
    {
        cv::Mat frame;

        // ====================================================
        // CAMERA FRAME
        // ====================================================

        try
        {
            camera >> frame;
        }
        catch (const cv::Exception& error)
        {
            std::cerr
                << "Camera frame error: "
                << error.what()
                << "\n";

            break;
        }

        if (frame.empty())
        {
            std::cerr
                << "WARNING: Empty camera frame.\n";

            continue;
        }

        // ====================================================
        // PALM DETECTION
        // ====================================================

        std::vector<std::vector<float>> palms;

        try
        {
            palms =
                palmDetector->detect(
                    frame
                );
        }
        catch (const Ort::Exception& error)
        {
            std::cerr
                << "Palm detection ONNX error: "
                << error.what()
                << "\n";

            continue;
        }
        catch (const cv::Exception& error)
        {
            std::cerr
                << "Palm detection OpenCV error: "
                << error.what()
                << "\n";

            continue;
        }
        catch (const std::exception& error)
        {
            std::cerr
                << "Palm detection error: "
                << error.what()
                << "\n";

            continue;
        }
        catch (...)
        {
            std::cerr
                << "Unknown palm detection error.\n";

            continue;
        }

        // ====================================================
        // FIRST VALID HAND
        // ====================================================

        bool handProcessed =
            false;

        for (const auto& palm : palms)
        {
            if (palm.size() < 18)
            {
                continue;
            }

            // =================================================
            // HAND POSE
            // =================================================

            std::vector<cv::Point3f> landmarks;

            try
            {
                landmarks =
                    handPose->detect(
                        frame,
                        palm
                    );
            }
            catch (const Ort::Exception& error)
            {
                std::cerr
                    << "Hand pose ONNX error: "
                    << error.what()
                    << "\n";

                continue;
            }
            catch (const cv::Exception& error)
            {
                std::cerr
                    << "Hand pose OpenCV error: "
                    << error.what()
                    << "\n";

                continue;
            }
            catch (const std::exception& error)
            {
                std::cerr
                    << "Hand pose error: "
                    << error.what()
                    << "\n";

                continue;
            }
            catch (...)
            {
                std::cerr
                    << "Unknown hand pose error.\n";

                continue;
            }

            // =================================================
            // REQUIRE 21 LANDMARKS
            // =================================================

            if (landmarks.size() != 21)
            {
                continue;
            }

            // =================================================
            // VALIDATE LANDMARKS
            // =================================================

            bool landmarksValid =
                true;

            for (const auto& point :
                landmarks)
            {
                if (!validLandmark(
                    point,
                    frame.cols,
                    frame.rows))
                {
                    landmarksValid =
                        false;

                    break;
                }
            }

            if (!landmarksValid)
            {
                continue;
            }

            // =================================================
            // PALM BOUNDING BOX
            // =================================================

            float x1 =
                std::clamp(
                    palm[0],
                    0.0f,
                    static_cast<float>(
                        frame.cols - 1
                        )
                );

            float y1 =
                std::clamp(
                    palm[1],
                    0.0f,
                    static_cast<float>(
                        frame.rows - 1
                        )
                );

            float x2 =
                std::clamp(
                    palm[2],
                    0.0f,
                    static_cast<float>(
                        frame.cols - 1
                        )
                );

            float y2 =
                std::clamp(
                    palm[3],
                    0.0f,
                    static_cast<float>(
                        frame.rows - 1
                        )
                );

            cv::rectangle(
                frame,
                cv::Point(
                    static_cast<int>(x1),
                    static_cast<int>(y1)
                ),
                cv::Point(
                    static_cast<int>(x2),
                    static_cast<int>(y2)
                ),
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );

            // =================================================
            // HAND SKELETON
            // =================================================

            const std::vector<
                std::pair<int, int>
            > connections =
            {
                // Thumb
                {0, 1},
                {1, 2},
                {2, 3},
                {3, 4},

                // Index
                {0, 5},
                {5, 6},
                {6, 7},
                {7, 8},

                // Middle
                {0, 9},
                {9, 10},
                {10, 11},
                {11, 12},

                // Ring
                {0, 13},
                {13, 14},
                {14, 15},
                {15, 16},

                // Little
                {0, 17},
                {17, 18},
                {18, 19},
                {19, 20},

                // Palm
                {5, 9},
                {9, 13},
                {13, 17}
            };

            for (const auto& connection :
                connections)
            {
                int a =
                    connection.first;

                int b =
                    connection.second;

                cv::Point pointA =
                    clampPoint(
                        landmarks[a],
                        frame.cols,
                        frame.rows
                    );

                cv::Point pointB =
                    clampPoint(
                        landmarks[b],
                        frame.cols,
                        frame.rows
                    );

                cv::line(
                    frame,
                    pointA,
                    pointB,
                    cv::Scalar(
                        255,
                        255,
                        0
                    ),
                    2
                );
            }

            // =================================================
            // DRAW 21 LANDMARKS
            // =================================================

            for (int i = 0;
                i < 21;
                ++i)
            {
                cv::Point point =
                    clampPoint(
                        landmarks[i],
                        frame.cols,
                        frame.rows
                    );

                cv::circle(
                    frame,
                    point,
                    5,
                    cv::Scalar(
                        255,
                        0,
                        0
                    ),
                    -1
                );

                cv::putText(
                    frame,
                    std::to_string(i),
                    point +
                    cv::Point(
                        6,
                        -6
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    cv::Scalar(
                        255,
                        255,
                        255
                    ),
                    1
                );
            }

            // =================================================
            // IMPORTANT LANDMARKS
            // =================================================

            const cv::Point3f& thumb =
                landmarks[4];

            const cv::Point3f& index =
                landmarks[8];

            const cv::Point3f& palm5 =
                landmarks[5];

            const cv::Point3f& palm17 =
                landmarks[17];

            // =================================================
            // PALM WIDTH
            // =================================================

            float palmWidth =
                static_cast<float>(
                    cv::norm(
                        cv::Point2f(
                            palm5.x,
                            palm5.y
                        )
                        -
                        cv::Point2f(
                            palm17.x,
                            palm17.y
                        )
                    )
                    );

            if (!std::isfinite(
                palmWidth) ||
                palmWidth < 1.0f)
            {
                palmWidth =
                    1.0f;
            }

            // =================================================
            // HAND GESTURE EXIT
            // =================================================
            // Close hand (fist), then open hand to exit.

            const cv::Point3f& wrist =
                landmarks[0];

            const cv::Point3f& indexTip =
                landmarks[8];

            const cv::Point3f& middleTip =
                landmarks[12];

            const cv::Point3f& ringTip =
                landmarks[16];

            const cv::Point3f& littleTip =
                landmarks[20];

            double indexOpen =
                cv::norm(
                    cv::Point2f(
                        indexTip.x,
                        indexTip.y
                    ) -
                    cv::Point2f(
                        wrist.x,
                        wrist.y
                    )
                );

            double middleOpen =
                cv::norm(
                    cv::Point2f(
                        middleTip.x,
                        middleTip.y
                    ) -
                    cv::Point2f(
                        wrist.x,
                        wrist.y
                    )
                );

            double ringOpen =
                cv::norm(
                    cv::Point2f(
                        ringTip.x,
                        ringTip.y
                    ) -
                    cv::Point2f(
                        wrist.x,
                        wrist.y
                    )
                );

            double littleOpen =
                cv::norm(
                    cv::Point2f(
                        littleTip.x,
                        littleTip.y
                    ) -
                    cv::Point2f(
                        wrist.x,
                        wrist.y
                    )
                );

            double averageFingerExtension =
                (indexOpen +
                    middleOpen +
                    ringOpen +
                    littleOpen) /
                4.0;

            double handOpenRatio =
                averageFingerExtension /
                static_cast<double>(palmWidth);

            if (!std::isfinite(handOpenRatio))
            {
                handOpenRatio = 0.0;
            }

            // Closed hand: fingertips are relatively close to the wrist.
            const bool fistDetected =
                handOpenRatio < 1.55;

            // Open hand: fingertips are extended away from the wrist.
            const bool openHandDetected =
                handOpenRatio > 2.00;

            // ---------------------------------------------
            // FIRST: DETECT CLOSED HAND
            // ---------------------------------------------

            if (!exitFistDetected)
            {
                if (fistDetected)
                {
                    fistFrames++;

                    if (fistFrames >= fistRequired)
                    {
                        exitFistDetected =
                            true;

                        openFrames =
                            0;

                        std::cout
                            << "Fist detected. Open hand to exit.\n";
                    }
                }
                else
                {
                    fistFrames =
                        0;
                }
            }

            // ---------------------------------------------
            // SECOND: DETECT OPEN HAND
            // ---------------------------------------------

            if (exitFistDetected)
            {
                if (openHandDetected)
                {
                    openFrames++;

                    if (openFrames >= openRequired)
                    {
                        std::cout
                            << "Exit gesture detected.\n";

                        running =
                            false;
                    }
                }
                else
                {
                    openFrames =
                        0;
                }
            }

            // =================================================
            // THUMB ↔ INDEX DISTANCE
            // =================================================

            float pinchDistance =
                static_cast<float>(
                    cv::norm(
                        cv::Point2f(
                            thumb.x,
                            thumb.y
                        )
                        -
                        cv::Point2f(
                            index.x,
                            index.y
                        )
                    )
                    );

            if (!std::isfinite(
                pinchDistance))
            {
                continue;
            }

            // =================================================
            // PINCH RATIO
            // =================================================

            float pinchRatio =
                pinchDistance /
                palmWidth;

            if (!std::isfinite(
                pinchRatio))
            {
                continue;
            }

            // =================================================
            // PINCH DETECTION
            // =================================================

            bool pinchDetected =
                pinchRatio <
                pinchStart;

            // =================================================
            // START PRECISION MODE
            // =================================================

            if (!precisionMode)
            {
                if (pinchDetected)
                {
                    pinchStartFrames++;

                    if (pinchStartFrames >=
                        startRequired)
                    {
                        precisionMode =
                            true;

                        clickArmed =
                            true;

                        pinchReleaseFrames =
                            0;
                    }
                }
                else
                {
                    pinchStartFrames =
                        0;
                }
            }

            // =================================================
            // PRECISION MODE
            // =================================================

            if (precisionMode)
            {
                double nx =
                    index.x /
                    static_cast<double>(
                        frame.cols
                        );

                double ny =
                    index.y /
                    static_cast<double>(
                        frame.rows
                        );

                nx =
                    std::clamp(
                        nx,
                        0.0,
                        1.0
                    );

                ny =
                    std::clamp(
                        ny,
                        0.0,
                        1.0
                    );

                // ---------------------------------------------
                // PRECISION MULTIPLIER
                // ---------------------------------------------

                const double precisionMultiplier =
                    1.5;

                double centeredX =
                    (nx - 0.5) *
                    precisionMultiplier +
                    0.5;

                double centeredY =
                    (ny - 0.5) *
                    precisionMultiplier +
                    0.5;

                centeredX =
                    std::clamp(
                        centeredX,
                        0.0,
                        1.0
                    );

                centeredY =
                    std::clamp(
                        centeredY,
                        0.0,
                        1.0
                    );

                double targetX =
                    centeredX *
                    screenWidth;

                double targetY =
                    centeredY *
                    screenHeight;

                // ---------------------------------------------
                // PRECISION DEAD ZONE
                // ---------------------------------------------

                double dx =
                    targetX -
                    smoothX;

                double dy =
                    targetY -
                    smoothY;

                double distance =
                    std::sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (distance >
                    deadZone)
                {
                    // -----------------------------------------
                    // VERY CLOSE
                    // -----------------------------------------

                    double sensitivity =
                        precisionSmoothing;

                    if (distance < 80.0)
                    {
                        sensitivity =
                            0.045;
                    }
                    else if (distance < 160.0)
                    {
                        sensitivity =
                            0.065;
                    }

                    smoothX +=
                        dx *
                        sensitivity;

                    smoothY +=
                        dy *
                        sensitivity;
                }

                // ---------------------------------------------
                // SAFE CURSOR LIMITS
                // ---------------------------------------------

                smoothX =
                    std::clamp(
                        smoothX,
                        0.0,
                        static_cast<double>(
                            screenWidth - 1
                            )
                    );

                smoothY =
                    std::clamp(
                        smoothY,
                        0.0,
                        static_cast<double>(
                            screenHeight - 1
                            )
                    );

                // ---------------------------------------------
                // SEND CURSOR ONLY WHEN NECESSARY
                // ---------------------------------------------

                double cursorDX =
                    smoothX -
                    lastCursorX;

                double cursorDY =
                    smoothY -
                    lastCursorY;

                if (std::abs(cursorDX) >=
                    minimumMove ||
                    std::abs(cursorDY) >=
                    minimumMove)
                {
                    SetCursorPos(
                        static_cast<int>(
                            smoothX
                            ),
                        static_cast<int>(
                            smoothY
                            )
                    );

                    lastCursorX =
                        smoothX;

                    lastCursorY =
                        smoothY;
                }

                // ---------------------------------------------
                // PINCH RELEASE
                // ---------------------------------------------

                if (pinchRatio >
                    pinchRelease)
                {
                    pinchReleaseFrames++;
                }
                else
                {
                    pinchReleaseFrames =
                        0;
                }

                // ---------------------------------------------
                // CLICK
                // ---------------------------------------------

                if (pinchReleaseFrames >=
                    releaseRequired &&
                    clickArmed)
                {
                    leftClick();

                    clickArmed =
                        false;

                    precisionMode =
                        false;

                    pinchStartFrames =
                        0;

                    pinchReleaseFrames =
                        0;
                }
            }

            // =================================================
            // NORMAL CURSOR MODE
            // =================================================

            if (!precisionMode)
            {
                double nx =
                    index.x /
                    static_cast<double>(
                        frame.cols
                        );

                double ny =
                    index.y /
                    static_cast<double>(
                        frame.rows
                        );

                nx =
                    std::clamp(
                        nx,
                        0.0,
                        1.0
                    );

                ny =
                    std::clamp(
                        ny,
                        0.0,
                        1.0
                    );

                // ---------------------------------------------
                // MOVEMENT ZONES
                // ---------------------------------------------
                //
                // Center = 1×
                // Middle = 1.5×
                // Outer = 2×
                //
                // ---------------------------------------------

                double multiplier =
                    1.0;

                if (nx < 0.125 ||
                    nx > 0.875 ||
                    ny < 0.125 ||
                    ny > 0.875)
                {
                    multiplier =
                        2.0;
                }
                else if (nx < 0.25 ||
                    nx > 0.75 ||
                    ny < 0.25 ||
                    ny > 0.75)
                {
                    multiplier =
                        1.5;
                }

                // ---------------------------------------------
                // TARGET POSITION
                // ---------------------------------------------

                double centeredX =
                    (nx - 0.5) *
                    multiplier +
                    0.5;

                double centeredY =
                    (ny - 0.5) *
                    multiplier +
                    0.5;

                centeredX =
                    std::clamp(
                        centeredX,
                        0.0,
                        1.0
                    );

                centeredY =
                    std::clamp(
                        centeredY,
                        0.0,
                        1.0
                    );

                double targetX =
                    centeredX *
                    screenWidth;

                double targetY =
                    centeredY *
                    screenHeight;

                // ---------------------------------------------
                // DIFFERENCE FROM CURRENT CURSOR
                // ---------------------------------------------

                double dx =
                    targetX -
                    smoothX;

                double dy =
                    targetY -
                    smoothY;

                double distance =
                    std::sqrt(
                        dx * dx +
                        dy * dy
                    );

                // ---------------------------------------------
                // ADAPTIVE SENSITIVITY
                // ---------------------------------------------

                if (distance >
                    deadZone)
                {
                    double sensitivity =
                        normalSmoothing;

                    // Far away = faster
                    if (distance > 500.0)
                    {
                        sensitivity =
                            0.30;
                    }
                    // Medium distance
                    else if (distance > 250.0)
                    {
                        sensitivity =
                            0.22;
                    }
                    // Near target = slower
                    else if (distance > 120.0)
                    {
                        sensitivity =
                            0.14;
                    }
                    // Very near target = very precise
                    else if (distance > 60.0)
                    {
                        sensitivity =
                            0.08;
                    }
                    else
                    {
                        sensitivity =
                            0.045;
                    }

                    smoothX +=
                        dx *
                        sensitivity;

                    smoothY +=
                        dy *
                        sensitivity;
                }

                // ---------------------------------------------
                // SAFE SCREEN LIMITS
                // ---------------------------------------------

                smoothX =
                    std::clamp(
                        smoothX,
                        0.0,
                        static_cast<double>(
                            screenWidth - 1
                            )
                    );

                smoothY =
                    std::clamp(
                        smoothY,
                        0.0,
                        static_cast<double>(
                            screenHeight - 1
                            )
                    );

                // ---------------------------------------------
                // ONLY MOVE IF SIGNIFICANT
                // ---------------------------------------------

                double cursorDX =
                    smoothX -
                    lastCursorX;

                double cursorDY =
                    smoothY -
                    lastCursorY;

                if (std::abs(cursorDX) >=
                    minimumMove ||
                    std::abs(cursorDY) >=
                    minimumMove)
                {
                    SetCursorPos(
                        static_cast<int>(
                            smoothX
                            ),
                        static_cast<int>(
                            smoothY
                            )
                    );

                    lastCursorX =
                        smoothX;

                    lastCursorY =
                        smoothY;
                }
            }

            // =================================================
            // PINCH VISUALIZATION
            // =================================================

            cv::Point thumbPoint =
                clampPoint(
                    thumb,
                    frame.cols,
                    frame.rows
                );

            cv::Point indexPoint =
                clampPoint(
                    index,
                    frame.cols,
                    frame.rows
                );

            cv::circle(
                frame,
                thumbPoint,
                9,
                cv::Scalar(
                    0,
                    255,
                    255
                ),
                2
            );

            cv::circle(
                frame,
                indexPoint,
                9,
                cv::Scalar(
                    0,
                    0,
                    255
                ),
                2
            );

            cv::line(
                frame,
                thumbPoint,
                indexPoint,
                cv::Scalar(
                    0,
                    255,
                    255
                ),
                2
            );

            // =================================================
            // STATUS
            // =================================================

            std::string status;

            if (precisionMode)
            {
                status =
                    "PRECISION MODE";
            }
            else
            {
                status =
                    "NORMAL CURSOR";
            }

            cv::putText(
                frame,
                status,
                cv::Point(
                    20,
                    35
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );

            // =================================================
            // PINCH RATIO
            // =================================================

            std::ostringstream ratioText;

            ratioText
                << "Pinch Ratio: "
                << std::fixed
                << std::setprecision(2)
                << pinchRatio;

            cv::putText(
                frame,
                ratioText.str(),
                cv::Point(
                    20,
                    65
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(
                    255,
                    255,
                    255
                ),
                1
            );

            // =================================================
            // TRACKING STATUS
            // =================================================

            cv::putText(
                frame,
                "Hand Tracking: ON",
                cv::Point(
                    20,
                    90
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                1
            );

            // =================================================
            // POINTING MODE
            // =================================================

            cv::putText(
                frame,
                "Smooth Precision Pointing: ON",
                cv::Point(
                    20,
                    115
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.50,
                cv::Scalar(
                    255,
                    255,
                    255
                ),
                1
            );

            handProcessed =
                true;

            // Process only first valid hand
            break;
        }

        // ====================================================
        // NO VALID HAND
        // ====================================================

        if (!handProcessed)
        {
            cv::putText(
                frame,
                "No valid hand detected",
                cv::Point(
                    20,
                    35
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(
                    0,
                    0,
                    255
                ),
                2
            );

            // Reset click state.
            // This prevents an old pinch from creating
            // an unexpected click when the hand returns.

            pinchStartFrames =
                0;

            pinchReleaseFrames =
                0;

            precisionMode =
                false;

            clickArmed =
                false;

            // Reset exit gesture if the hand is lost.
            fistFrames =
                0;

            openFrames =
                0;

            exitFistDetected =
                false;
        }

        // ====================================================
        // DISPLAY
        // ====================================================

        try
        {
            cv::imshow(
                windowName,
                frame
            );
        }
        catch (const cv::Exception& error)
        {
            std::cerr
                << "Display error: "
                << error.what()
                << "\n";

            break;
        }

        // ====================================================
        // KEYBOARD
        // ====================================================
        // Keyboard input is kept for OpenCV window processing.
        // Program exit is now controlled by: FIST -> OPEN HAND.

        cv::waitKey(1);
    }

    // ========================================================
    // CLEANUP
    // ========================================================

    std::cout
        << "\nClosing Gesture AI Mouse...\n";

    delete palmDetector;
    delete handPose;

    camera.release();

    cv::destroyAllWindows();

    std::cout
        << "Program closed safely.\n";

    return 0;
}