/**
 * CIFAR-10 ONNX Runtime C++ Inference
 *
 * Compile:
 *   mkdir build && cd build
 *   cmake ..
 *   make
 *
 * Run:
 *   ./cifar_infer_onnx ../models/cifar10_resnet18.onnx ../data/test.png
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include <opencv2/opencv.hpp>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// CIFAR-10 classes
const std::vector<std::string> CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Normalization parameters
const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
const float STD[3] = {0.2023f, 0.1994f, 0.2010f};


/**
 * Preprocess image for CIFAR-10 ONNX model
 */
std::vector<float> preprocess(const std::string& img_path) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        throw std::runtime_error("Cannot read image: " + img_path);
    }

    cv::resize(img, img, cv::Size(32, 32));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(img_rgb, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - MEAN[c]) / STD[c];
    }

    cv::Mat chw;
    cv::merge(channels, chw);

    std::vector<float> input(1 * 3 * 32 * 32);
    std::memcpy(input.data(), chw.ptr<float>(), chw.total() * sizeof(float));

    return input;
}


/**
 * Print top-5 predictions
 */
void printTop5(const float* probs, int num_classes) {
    std::vector<std::pair<int, float>> pairs;
    for (int i = 0; i < num_classes; i++) {
        pairs.push_back(std::make_pair(i, probs[i]));
    }

    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });

    std::cout << std::endl;
    std::cout << "Top-5 predictions:" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < 5; i++) {
        int idx = pairs[i].first;
        float prob = pairs[i].second * 100.0f;

        int bar_len = static_cast<int>(prob / 5.0f);
        std::string bar(bar_len, '#') + std::string(20 - bar_len, '-');

        std::cout << "  " << i + 1 << ". " << std::setw(12) << CLASSES[idx]
                  << ": " << std::setw(6) << prob << "%  " << bar << std::endl;
    }
}


void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <model.onnx> <image.png>" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog << " models/cifar10_resnet18.onnx data/test.png" << std::endl;
}


int main(int argc, char* argv[]) {
    std::cout << std::endl;
    std::cout << "=== CIFAR-10 ONNX Runtime C++ Inference ===" << std::endl;
    std::cout << std::endl;

    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    // Check files exist
    FILE* f = std::fopen(modelPath.c_str(), "rb");
    if (!f) {
        std::cerr << "Error: Model file not found: " << modelPath << std::endl;
        return 1;
    }
    std::fclose(f);

    f = std::fopen(imagePath.c_str(), "rb");
    if (!f) {
        std::cerr << "Error: Image file not found: " << imagePath << std::endl;
        return 1;
    }
    std::fclose(f);

    try {
        std::cout << "Loading model: " << modelPath << std::endl;
        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

#ifdef __APPLE__
        if (Ort::CoreMLProviderHasCoreMLExecutionProvider()) {
            session_options.AppendExecutionProvider_CoreML(
                Ort::CoreMLExecutionProviderOptions{0});
            std::cout << "   Using CoreML provider" << std::endl;
        }
#endif

        Ort::Session session(env, modelPath.c_str(), session_options);

        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        std::cout << "   Inputs: " << num_inputs << ", Outputs: " << num_outputs << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        std::string input_name = session.GetInputNameAllocated(0, allocator).get();
        std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
        std::cout << "   Input: " << input_name << ", Output: " << output_name << std::endl;
        std::cout << "   Model loaded successfully" << std::endl;

        std::cout << "Processing image: " << imagePath << std::endl;
        std::vector<float> input_data = preprocess(imagePath);
        std::cout << "   Input shape: (1, 3, 32, 32)" << std::endl;

        std::vector<int64_t> input_shape = {1, 3, 32, 32};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        std::cout << "Running inference..." << std::endl;
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(input_tensor));

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name, inputs.data(), 1,
            &output_name, 1
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_classes = static_cast<int>(output_shape[1]);
        std::cout << "   Output shape: (" << output_shape[0] << ", " << num_classes << ")" << std::endl;

        // Softmax with numerical stability
        float max_val = output_data[0];
        for (int i = 1; i < num_classes; i++) {
            if (output_data[i] > max_val) max_val = output_data[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output_data[i] = std::exp(output_data[i] - max_val);
            sum += output_data[i];
        }
        for (int i = 0; i < num_classes; i++) {
            output_data[i] /= sum;
        }

        // Find prediction
        int pred_class = 0;
        float max_prob = output_data[0];
        for (int i = 1; i < num_classes; i++) {
            if (output_data[i] > max_prob) {
                max_prob = output_data[i];
                pred_class = i;
            }
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Prediction: " << CLASSES[pred_class] << std::endl;
        std::cout << "Confidence: " << (max_prob * 100.0f) << "%" << std::endl;
        std::cout << "========================================" << std::endl;

        printTop5(output_data, num_classes);

        std::cout << std::endl;
        std::cout << "Done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
