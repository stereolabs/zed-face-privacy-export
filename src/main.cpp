///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <math.h>

#define ENABLE_GUI 1

// Flag to enable blur instead of black fill for privacy
#define ENABLE_BLUR_PRIVACY 1

// ZED includes
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;
bool is_playback = false;

bool isGuiAvailable() {
    const char* display = getenv("DISPLAY");
    return (display != nullptr && strlen(display) > 0);
}

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "") {
    cout << "[Sample] ";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

inline cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1;
            break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2;
            break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3;
            break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4;
            break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1;
            break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2;
            break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3;
            break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4;
            break;
        default: break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

template<typename T>
inline cv::Point2f cvt(T pt) {
    return cv::Point2f(pt.x, pt.y);
}

inline sl::float2 getImagePosition(std::vector<sl::uint2> &bounding_box_image) {
    sl::float2 position;
    position.x = (bounding_box_image[0].x + (bounding_box_image[2].x - bounding_box_image[0].x)*0.5f);
    position.y = (bounding_box_image[0].y + (bounding_box_image[2].y - bounding_box_image[0].y)*0.5f);
    return position;
}

void render_2D(cv::Mat &left_display, std::vector<sl::ObjectData> &objects) {
    // render bounding boxes and mask if available
    for (auto& obj : objects) {
        // Display Image scaled bounding box 2D
        if (obj.bounding_box_2d.empty()) {
            continue;
        }

        cv::Point top_left_corner = cvt(obj.bounding_box_2d[0]);
        cv::Point top_right_corner = cvt(obj.bounding_box_2d[1]);
        cv::Point bottom_right_corner = cvt(obj.bounding_box_2d[2]);
        cv::Point bottom_left_corner = cvt(obj.bounding_box_2d[3]);

        // scaled ROI
        cv::Rect roi(top_left_corner, bottom_right_corner);

        // Ensure ROI is within image bounds
        roi &= cv::Rect(0, 0, left_display.cols, left_display.rows);

        if (roi.width > 0 && roi.height > 0) {
#if ENABLE_BLUR_PRIVACY
            // Apply Gaussian blur for privacy instead of black fill
            // Scale blur intensity based on bounding box size (larger = closer = more blur)
            int bbox_area = roi.width * roi.height;
            int base_kernel_size = 25;
            int max_kernel_size = 101;
            int min_kernel_size = 15;

            // Scale kernel size based on bounding box area
            float area_ratio = static_cast<float> (bbox_area) / (left_display.cols * left_display.rows);
            int kernel_size = base_kernel_size + static_cast<int> (area_ratio * 50000);
            kernel_size = std::max(min_kernel_size, std::min(max_kernel_size, kernel_size));

            // Ensure kernel size is odd
            if (kernel_size % 2 == 0) kernel_size++;

            cv::Mat roi_region = left_display(roi);
            cv::Mat blurred_roi;
            cv::GaussianBlur(roi_region, blurred_roi, cv::Size(kernel_size, kernel_size), 0);
            blurred_roi.copyTo(left_display(roi));
#else // black square
            // Use isInit() to check if mask is available
            left_display(roi).setTo(cv::Scalar(0, 0, 0, 0));
#endif
        }
    }
}

void printProgressBar(int current, int total, int barWidth = 50) {
    float progress = float(current) / total;
    int pos = int(barWidth * progress);
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%\r";
    std::cout.flush();
}

void parseArgs(int argc, char **argv, InitParameters& param, std::string& outputFile) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input.svo> [output_file]" << std::endl;
        std::exit(1);
    }

    std::string inputFile = argv[1];
    if (inputFile.find(".svo") == std::string::npos && inputFile.find(".svo2") == std::string::npos) {
        std::cout << "[Error] First argument must be an SVO file (*.svo or *.svo2)" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input.svo|input.svo2> [output_file]" << std::endl;
        std::exit(1);
    }

    // SVO input mode
    param.input.setFromSVOFile(inputFile.c_str());
    is_playback = true;
    std::cout << "[Sample] Using SVO File input: " << inputFile << std::endl;

    // Optional output file parameter
    if (argc > 2) {
        outputFile = argv[2];
        std::cout << "[Sample] Output file set to: " << outputFile << std::endl;
    }
}

int main(int argc, char **argv) {

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    //init_parameters.depth_mode = DEPTH_MODE::NEURAL_PLUS;
    init_parameters.depth_maximum_distance = 50.0f * 1000.0f;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.sdk_verbose = 1;
    std::string output_file = "svo_blurred_faces.avi";

    parseArgs(argc, argv, init_parameters, output_file);

    // Check if GUI is available
    bool gui_available = isGuiAvailable();
    if (!gui_available) {
        std::cout << "[Info] No GUI detected (DISPLAY not set), disabling GUI features" << std::endl;
    }

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;
    PositionalTrackingParameters positional_tracking_parameters;
    // If the camera is static in space, enabling this settings below provides better depth quality and faster computation
    positional_tracking_parameters.set_as_static = true;
    zed.enablePositionalTracking(positional_tracking_parameters);

    print("Object Detection: Loading Module...");
    // Define the Objects detection module parameters
    ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = 0;
    detection_parameters.enable_segmentation = false;
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::PERSON_HEAD_BOX_ACCURATE;

    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    // Detection runtime parameters
    // default detection threshold, apply to all object class
    int detection_confidence = 15;
    ObjectDetectionRuntimeParameters detection_parameters_rt(detection_confidence);

    // Detection output
    bool quit = false;
    cv::Mat cv_img;
    sl::Mat sl_img;

    int width = camera_config.resolution.width;
    int height = camera_config.resolution.height;
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    //int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');  // Changed to MJPG for better compatibility
    double fps = camera_config.fps;
    cv::VideoWriter writer;
    
    std::cout << "Video specs: " << width << "x" << height << " @ " << fps << " fps" << std::endl;
        
    writer.open(output_file, fourcc, fps, cv::Size(width, height), 1);
        
    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file for write.\n";
        return -1;
    }

#if ENABLE_GUI
    string window_name = "ZED";
    char key = ' ';
    
    if (gui_available) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::createTrackbar("Confidence", window_name, &detection_confidence, 100);
    }
#endif

    Objects objects;
    int i = 0;
    int num_frames = zed.getSVONumberOfFrames();
    while (!quit && zed.grab() <= ERROR_CODE::SUCCESS) {
        // update confidence threshold based on TrackBar
        detection_parameters_rt.detection_confidence_threshold = detection_confidence;

        returned_state = zed.retrieveObjects(objects, detection_parameters_rt);
        if (returned_state <= ERROR_CODE::SUCCESS) {

            zed.retrieveImage(sl_img, VIEW::LEFT, MEM::CPU);
            cv_img = slMat2cvMat(sl_img);
            
            // Convert BGRA to BGR if necessary
            if (cv_img.channels() == 4) {
                cv::cvtColor(cv_img, cv_img, cv::COLOR_BGRA2BGR);
            }
            
            render_2D(cv_img, objects.object_list);
            
            // Verify frame properties before writing
            if (cv_img.rows == height && cv_img.cols == width && cv_img.channels() == 3) {
                writer.write(cv_img);
            } else {
                std::cerr << "Frame size mismatch: " << cv_img.cols << "x" << cv_img.rows 
                         << " channels: " << cv_img.channels() << std::endl;
            }

            printProgressBar(i++, num_frames);

#if ENABLE_GUI
            if (gui_available) {
                cv::imshow(window_name, cv_img);
                key = cv::waitKey(10);
                if (key == 'q') quit = true;
            }
#endif
        }

        if (is_playback && zed.getSVOPosition() == zed.getSVONumberOfFrames())
            quit = true;
    }

    writer.release();
    std::cout << "\nVideo written to " << output_file << std::endl;

    zed.disableObjectDetection();
    zed.close();
    return EXIT_SUCCESS;
}



