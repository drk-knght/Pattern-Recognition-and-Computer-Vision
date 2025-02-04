// src/q7.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

// Structure for holding a match result.
struct Match
{
    std::string filename;
    double distance;
};

// Helper function to get the shape of a blob represented by cv::Mat.
std::vector<int> getBlobShape(const cv::Mat &blob)
{
    std::vector<int> shape;
    for (int i = 0; i < blob.dims; ++i)
    {
        shape.push_back(blob.size[i]);
    }
    return shape;
}

// Computes a simple color histogram feature vector for an image.
// The histogram is computed per channel (8 bins per channel) and normalized.
std::vector<double> computeColorHistogram(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to computeColorHistogram." << std::endl;
        return {};
    }
    cv::Mat img;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = image;
    }
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    int histSize = 8;         // number of bins per channel
    float range[] = {0, 256}; // range of pixel values
    const float *histRange = {range};
    bool uniform = true, accumulate = false;
    std::vector<double> histFeature;
    // Compute histogram for each channel over the whole image.
    for (size_t i = 0; i < channels.size(); i++)
    {
        cv::Mat hist;
        cv::calcHist(&channels[i], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
        hist /= static_cast<double>(img.total());
        for (int bin = 0; bin < histSize; bin++)
        {
            histFeature.push_back(hist.at<float>(bin));
        }
    }
    return histFeature;
}

// Computes the sum-of-squares (SSD) distance between two feature vectors.
double ssd_distance(const std::vector<double> &v1, const std::vector<double> &v2)
{
    if (v1.size() != v2.size())
    {
        std::cerr << "Error: Feature vector sizes do not match in ssd_distance." << std::endl;
        return std::numeric_limits<double>::max();
    }
    double dist = 0.0;
    for (size_t i = 0; i < v1.size(); i++)
    {
        double diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return dist;
}

// Computes the cosine distance between two feature vectors.
double cosine_distance(const std::vector<double> &v1, const std::vector<double> &v2)
{
    if (v1.size() != v2.size())
    {
        std::cerr << "Error: Feature vector sizes do not match in cosine_distance." << std::endl;
        return 1.0;
    }
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < v1.size(); i++)
    {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0)
        return 1.0;
    double cosineSim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
    return 1.0 - cosineSim;
}

// Computes a depth map for a given image using a DNN model.
// The model is assumed to output a single-channel map (with shape 1x1xH'xW').
// The result is resized to the original image dimensions and normalized to [0,1].
cv::Mat computeDepthMap(const cv::Mat &image, cv::dnn::Net &depthNet)
{
    // Preprocess: resize to the network's expected size (e.g., 384x384 for MiDaS v2 small).
    cv::Size inputSize(384, 384);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, inputSize, cv::Scalar(0, 0, 0), true, false);
    depthNet.setInput(blob);
    cv::Mat depth = depthNet.forward();

    // The network output is 4D: 1 x 1 x H' x W'. Reshape it.
    cv::Mat depthMap(depth.size[2], depth.size[3], CV_32F, depth.ptr<float>());
    depthMap = depthMap.clone(); // clone to ensure data persists

    // Resize depth map to the original image size.
    cv::Mat depthMapResized;
    cv::resize(depthMap, depthMapResized, image.size(), 0, 0, cv::INTER_CUBIC);

    // Normalize the depth map to [0, 1].
    double minVal, maxVal;
    cv::minMaxLoc(depthMapResized, &minVal, &maxVal);
    cv::Mat depthMapNormalized;
    if (maxVal - minVal > 0)
        depthMapNormalized = (depthMapResized - minVal) / (maxVal - minVal);
    else
        depthMapNormalized = depthMapResized.clone();

    return depthMapNormalized;
}

// Computes the color histogram feature vector for an image
// but only over pixels with a depth value lower than depthThreshold.
// This function uses a depth estimation model to compute the depth map
// and then creates a mask. If no pixel is below the threshold, it falls back
// to computing the histogram over the whole image.
std::vector<double> computeColorHistogramWithDepth(const cv::Mat &image, cv::dnn::Net &depthNet, double depthThreshold)
{
    if (image.empty())
    {
        std::cerr << "Error: Empty image provided to computeColorHistogramWithDepth." << std::endl;
        return {};
    }
    cv::Mat img;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = image;
    }

    // Compute the depth map.
    cv::Mat depthMap = computeDepthMap(img, depthNet);

    // Create a mask: include only pixels with depth less than depthThreshold.
    // (Assumes that after normalization lower values indicate "closer.")
    cv::Mat mask;
    cv::compare(depthMap, depthThreshold, mask, cv::CMP_LT); // mask=255 where depthMap < depthThreshold

    // Compute the total number of pixels passing the threshold.
    double pixelCount = cv::countNonZero(mask);
    if (pixelCount == 0)
    {
        std::cerr << "Warning: No pixels under depth threshold, using entire image for histogram." << std::endl;
        return computeColorHistogram(image);
    }

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    int histSize = 8;
    float range[] = {0, 256};
    const float *histRange = {range};
    bool uniform = true, accumulate = false;
    std::vector<double> histFeature;
    // Compute histogram per channel on the masked pixels.
    for (size_t i = 0; i < channels.size(); i++)
    {
        cv::Mat hist;
        cv::calcHist(&channels[i], 1, 0, mask, hist, 1, &histSize, &histRange, uniform, accumulate);
        hist /= pixelCount; // normalize by the number of valid (masked) pixels
        for (int bin = 0; bin < histSize; bin++)
        {
            histFeature.push_back(hist.at<float>(bin));
        }
    }
    return histFeature;
}

// Helper to print usage information.
void print_usage(const std::string &prog_name)
{
    std::cout << "Usage: " << prog_name
              << " --target TARGET_IMAGE --dir IMAGE_DIRECTORY [--dist DISTANCE_METRIC] [--top N]\n"
              << "       [--depth-threshold THRESHOLD] [--depth-model MODEL_PATH]\n"
              << "  --target: path to the target image\n"
              << "  --dir   : directory containing database images\n"
              << "  --dist  : distance metric to use ('ssd' or 'cosine'; default: ssd)\n"
              << "  --top   : number of top matches to return (default: 3)\n"
              << "  --depth-threshold: relative depth threshold in [0,1] to filter pixels (optional)\n"
              << "  --depth-model: path to the ONNX model for depth estimation (default: midas.onnx) (optional)\n";
}

// Checks if the given file name has a valid image extension.
bool isValidImageFile(const std::string &filename)
{
    auto pos = filename.find_last_of("/\\");
    std::string basename = (pos == std::string::npos) ? filename : filename.substr(pos + 1);
    if (!basename.empty() && basename[0] == '.')
        return false;
    auto dotPos = basename.find_last_of('.');
    if (dotPos == std::string::npos)
        return false;
    std::string ext = basename.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    static const std::set<std::string> validExtensions = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"};
    return (validExtensions.find(ext) != validExtensions.end());
}

// Example: Custom layer for the patch embedding operator (dummy passthrough)
class CustomPatchEmbedLayer : public cv::dnn::Layer
{
public:
    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams &params)
    {
        return cv::Ptr<cv::dnn::Layer>(new CustomPatchEmbedLayer(params));
    }
    CustomPatchEmbedLayer(cv::dnn::LayerParams &params) : cv::dnn::Layer(params)
    {
        // Any initialization here if needed
    }
    virtual bool getMemoryShapes(const std::vector<std::vector<int>> &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int>> &outputs,
                                 std::vector<std::vector<int>> &internals) const override
    {
        // For now, simply copy the input shape to the output.
        outputs = inputs;
        return false;
    }
    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) override
    {
        std::vector<cv::Mat> inputs;
        inputs_arr.getMatVector(inputs);
        std::vector<cv::Mat> outputs;
        outputs_arr.getMatVector(outputs);
        if (!inputs.empty() && !outputs.empty())
        {
            // Dummy passthrough: output equals input.
            outputs[0] = inputs[0];
        }
    }
};

// Example: Custom layer for aten_expand (dummy passthrough implementation)
class CustomAtenExpandLayer : public cv::dnn::Layer
{
public:
    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams &params)
    {
        return cv::Ptr<cv::dnn::Layer>(new CustomAtenExpandLayer(params));
    }
    CustomAtenExpandLayer(cv::dnn::LayerParams &params) : cv::dnn::Layer(params)
    {
        // Initialization if needed
    }
    virtual bool getMemoryShapes(const std::vector<std::vector<int>> &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int>> &outputs,
                                 std::vector<std::vector<int>> &internals) const override
    {
        // In practice, aten_expand should compute the output shape based on the expansion parameters.
        // For a dummy implementation we simply:
        // - If a second input (target shape) is provided, use it;
        // - Otherwise, pass through the input shape.
        if (inputs.size() > 1 && !inputs[1].empty())
        {
            outputs.push_back(inputs[1]);
        }
        else
        {
            outputs = inputs;
        }
        return false;
    }
    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) override
    {
        std::vector<cv::Mat> inputs;
        inputs_arr.getMatVector(inputs);
        std::vector<cv::Mat> outputs;
        outputs_arr.getMatVector(outputs);
        if (!inputs.empty() && !outputs.empty())
        {
            // Dummy passthrough: simply copy the input data.
            outputs[0] = inputs[0].clone();
        }
    }
};

// Updated CustomNegLayer to ensure that inputs are never empty.
class CustomNegLayer : public cv::dnn::Layer
{
public:
    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams &params)
    {
        return cv::Ptr<cv::dnn::Layer>(new CustomNegLayer(params));
    }
    CustomNegLayer(cv::dnn::LayerParams &params) : cv::dnn::Layer(params) {}

    // Override getMemoryShapes() to always return a valid (non-empty) output shape.
    virtual bool getMemoryShapes(const std::vector<std::vector<int>> &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int>> &outputs,
                                 std::vector<std::vector<int>> &internals) const override
    {
        std::vector<int> shape;
        // If inputs is empty or its first element is empty, provide a default shape.
        if (inputs.empty() || inputs[0].empty())
        {
            shape = {1};
        }
        else
        {
            shape = inputs[0];
        }
        outputs.push_back(shape);
        return false;
    }

    // Override forward() to ensure we have at least one input.
    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) override
    {
        std::vector<cv::Mat> inputMats;
        inputs_arr.getMatVector(inputMats);
        // If no input is provided, create a dummy 1x1 float Mat.
        if (inputMats.empty())
        {
            inputMats.push_back(cv::Mat::zeros(1, 1, CV_32F));
        }

        std::vector<cv::Mat> outputMats;
        outputs_arr.getMatVector(outputMats);
        if (outputMats.empty())
            return;

        cv::Mat negated;
        cv::subtract(cv::Scalar::all(0), inputMats[0], negated);
        outputMats[0] = negated;
    }
};

// -----------------------------------------------------------------------------
// DNN feature: ResNet18 embedding extraction
//
// This function reads an image, preprocesses it (resizes, scales, subtracts mean),
// and then feeds it through the network to obtain the feature embedding.
//
// The embedding is obtained from a specific layer (here "onnx_node!resnetv22_flatten0_reshape0").
// Adjust the layer name or preprocessing if using a different model.
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug = 0)
{
    const int targetSize = 224;
    cv::Mat blob;
    // Preprocess: scale input, resize to targetSize x targetSize, subtract mean.
    cv::dnn::blobFromImage(src, blob, (1.0 / 255.0) * (1 / 0.226), cv::Size(targetSize, targetSize), cv::Scalar(124, 116, 104), true, false, CV_32F);
    net.setInput(blob);
    // Adjust the layer name below as required by your ResNet18 model.
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    if (debug)
    {
        cv::imshow("src", src);
        std::cout << "DNN embedding:" << std::endl
                  << embedding << std::endl;
        cv::waitKey(0);
    }
    return 0;
}

// Helper: Convert a cv::Mat (assumed type CV_32F and one row) to std::vector<double>
std::vector<double> matToVector(const cv::Mat &mat)
{
    std::vector<double> v;
    if (mat.empty())
        return v;
    // Assume mat is a row vector or a flattened result.
    for (int i = 0; i < mat.total(); i++)
    {
        v.push_back(mat.at<float>(i));
    }
    return v;
}

// -----------------------------------------------------------------------------
// Main function
//
// Expected command-line arguments:
//   --target <target_image_path>
//   --dir <database_directory>
//   [--dist <distance_metric> ('ssd' or 'cosine'; default: ssd)]
//   [--top <N>         (number of top matches; default: 3)]
//   [--embedding-model <path_to_resnet18_onnx> (default: resnet18.onnx)]
//
int main(int argc, char *argv[])
{
    // *** IMPORTANT: Register the custom layers BEFORE loading any ONNX model ***
    cv::dnn::LayerFactory::registerLayer(
        "pkg.depth_anything_v2.depth_anything_v2_dinov2_layers_patch_embed_PatchEmbed_pretrained_patch_embed_1",
        CustomPatchEmbedLayer::create);
    cv::dnn::LayerFactory::registerLayer(
        "pkg.onnxscript.torch_lib.aten_expand",
        CustomAtenExpandLayer::create);
    cv::dnn::LayerFactory::registerLayer(
        "ai.onnx.Neg",
        CustomNegLayer::create);

    std::string targetImagePath;
    std::string dbDirectory;
    std::string distanceMetric = "ssd";
    int top_n = 3;
    std::string embeddingModelPath = "resnet18.onnx"; // default embedding model

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " --target TARGET_IMAGE --dir IMAGE_DIRECTORY [--dist DISTANCE_METRIC] [--top N] [--embedding-model MODEL_PATH]" << std::endl;
        std::cerr << "  --target: path to the target image" << std::endl;
        std::cerr << "  --dir   : directory containing database images" << std::endl;
        std::cerr << "  --dist  : distance metric ('ssd' or 'cosine'; default: ssd)" << std::endl;
        std::cerr << "  --top   : number of top matches to return (default: 3)" << std::endl;
        std::cerr << "  --embedding-model: path to the ResNet ONNX model for embedding (default: resnet18.onnx)" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--target" && i + 1 < argc)
        {
            targetImagePath = argv[++i];
        }
        else if (arg == "--dir" && i + 1 < argc)
        {
            dbDirectory = argv[++i];
        }
        else if (arg == "--dist" && i + 1 < argc)
        {
            distanceMetric = argv[++i];
        }
        else if (arg == "--top" && i + 1 < argc)
        {
            top_n = std::stoi(argv[++i]);
        }
        else if (arg == "--embedding-model" && i + 1 < argc)
        {
            embeddingModelPath = argv[++i];
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " --target TARGET_IMAGE --dir IMAGE_DIRECTORY [--dist DISTANCE_METRIC] [--top N] [--embedding-model MODEL_PATH]" << std::endl;
            return 1;
        }
    }

    // Load the target image.
    cv::Mat targetImg = cv::imread(targetImagePath);
    if (targetImg.empty())
    {
        std::cerr << "Error: Could not open or find target image " << targetImagePath << std::endl;
        return 1;
    }

    // Compute the color histogram for the target image.
    std::vector<double> targetHist = computeColorHistogram(targetImg);

    // Load the embedding network.
    cv::dnn::Net embedNet = cv::dnn::readNetFromONNX(embeddingModelPath);
    if (embedNet.empty())
    {
        std::cerr << "Error: Could not load embedding model from " << embeddingModelPath << std::endl;
        return 1;
    }

    // Compute the DNN embedding for the target image.
    cv::Mat targetEmbeddingMat;
    getEmbedding(targetImg, targetEmbeddingMat, embedNet, 0);
    std::vector<double> targetEmbedVec = matToVector(targetEmbeddingMat);

    // Iterate over all images in the provided directory.
    std::vector<cv::String> imageFiles;
    cv::glob(dbDirectory + "/*.*", imageFiles, false);
    if (imageFiles.empty())
    {
        std::cerr << "No images found in directory " << dbDirectory << std::endl;
        return 1;
    }

    // Extract the basename of the target image to skip it in the comparisons.
    size_t pos = targetImagePath.find_last_of("/\\");
    std::string targetBasename = (pos == std::string::npos) ? targetImagePath : targetImagePath.substr(pos + 1);

    std::vector<Match> matches;

    // Weights for the two feature distances.
    double weightHist = 1.0;
    double weightEmbed = 1.0;

    for (const auto &path : imageFiles)
    {
        size_t pos2 = path.find_last_of("/\\");
        std::string basename = (pos2 == std::string::npos) ? path : path.substr(pos2 + 1);
        if (basename == targetBasename)
            continue;

        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            std::cerr << "Warning: Could not read image " << path << std::endl;
            continue;
        }
        // Compute features for the current image.
        std::vector<double> currHist = computeColorHistogram(img);
        cv::Mat currEmbeddingMat;
        getEmbedding(img, currEmbeddingMat, embedNet, 0);
        std::vector<double> currEmbedVec = matToVector(currEmbeddingMat);

        // Compute distances.
        double dHist = ssd_distance(targetHist, currHist);
        double dEmbed = cosine_distance(targetEmbedVec, currEmbedVec);
        double compositeDistance = weightHist * dHist + weightEmbed * dEmbed;

        matches.push_back({path, compositeDistance});
    }

    if (matches.empty())
    {
        std::cerr << "No valid images found or features computed." << std::endl;
        return 1;
    }

    // Sort the matches (smaller composite distance means more similar).
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
              { return a.distance < b.distance; });

    // Print out the top N matches.
    std::cout << "\nTop " << top_n << " matches for target image '" << targetImagePath << "':\n";
    for (int i = 0; i < top_n && i < static_cast<int>(matches.size()); i++)
    {
        std::cout << (i + 1) << ". " << matches[i].filename << " (composite distance: " << matches[i].distance << ")\n";
    }

    return 0;
}