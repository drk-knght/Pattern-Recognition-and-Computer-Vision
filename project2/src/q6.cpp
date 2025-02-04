#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <set> // For allowed file extensions.
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Helper function to check if a file has a valid image extension.
// It skips files like .DS_Store as well as any file that doesn't have
// an allowed image extension (e.g., jpg, jpeg, png, bmp, tiff, tif).
bool isValidImageFile(const std::string &filename)
{
    // Extract the basename from the full file path.
    auto pos = filename.find_last_of("/\\");
    std::string basename = (pos == std::string::npos) ? filename : filename.substr(pos + 1);

    // Skip hidden files (those starting with a dot; e.g., .DS_Store).
    if (!basename.empty() && basename[0] == '.')
        return false;

    // Check if the basename contains an extension.
    auto dotPos = basename.find_last_of('.');
    if (dotPos == std::string::npos)
        return false; // No extension found, so skip file.

    std::string ext = basename.substr(dotPos + 1);
    // Convert extension to lower-case.
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Define allowed image file extensions.
    static const std::set<std::string> validExtensions = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"};
    return validExtensions.find(ext) != validExtensions.end();
}

// Helper structure for holding a match result.
struct Match
{
    std::string filename;
    double distance;
};

// Compute DNN features using a pre-trained ONNX model.
// Note: the model is assumed to expect a 224x224 BGR image with a specific mean subtraction.
std::vector<double> computeDNNFeatures(const cv::Mat &img, cv::dnn::Net &net)
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    // Preprocess: create blob (here mean values are those typically used for ResNet)
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0, cv::Size(224, 224),
                                          cv::Scalar(103.94, 116.78, 123.68), // B, G, R means
                                          false, false);
    net.setInput(blob);
    cv::Mat output = net.forward();
    // Flatten the output to a vector (assuming output is CV_32F)
    std::vector<float> featureFloat;
    featureFloat.assign((float *)output.datastart, (float *)output.dataend);
    std::vector<double> feature(featureFloat.begin(), featureFloat.end());
    return feature;
}

// Computes the sum-of-squares (SSD) (squared Euclidean) distance between two feature vectors.
double ssd_distance(const std::vector<double> &v1, const std::vector<double> &v2)
{
    if (v1.size() != v2.size())
    {
        std::cerr << "Feature vector sizes do not match." << std::endl;
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

// Helper function to print usage instructions.
void printUsage(const std::string &progName)
{
    std::cout << "Usage: " << progName
              << " --target TARGET_IMAGE --db IMAGE_DIRECTORY [--top N] [--dnnmodel PATH]\n"
              << "  --target : The target image filename\n"
              << "  --db     : Directory containing the image database to search\n"
              << "  --top    : Number of top matches to return (default: 3)\n"
              << "  --dnnmodel: Path to the ONNX DNN model (default: resnet50.onnx)\n";
}

int main(int argc, char *argv[])
{
    // Parameters.
    std::string targetImageFile;
    std::string dbDir;
    int topN = 3;
    std::string dnnModelPath = "resnet50.onnx";

    // Process command-line arguments.
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--target" && i + 1 < argc)
        {
            targetImageFile = argv[++i];
        }
        else if (arg == "--db" && i + 1 < argc)
        {
            dbDir = argv[++i];
        }
        else if (arg == "--top" && i + 1 < argc)
        {
            topN = std::stoi(argv[++i]);
        }
        else if (arg == "--dnnmodel" && i + 1 < argc)
        {
            dnnModelPath = argv[++i];
        }
        else if (arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (targetImageFile.empty() || dbDir.empty())
    {
        std::cerr << "Error: Both --target and --db parameters must be provided." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Load the target image.
    cv::Mat targetImg = cv::imread(targetImageFile);
    if (targetImg.empty())
    {
        std::cerr << "Error: Could not load target image " << targetImageFile << std::endl;
        return 1;
    }

    // Load the DNN model.
    cv::dnn::Net net = cv::dnn::readNetFromONNX(dnnModelPath);
    if (net.empty())
    {
        std::cerr << "Error: Could not load DNN model from " << dnnModelPath << std::endl;
        return 1;
    }

    // Compute DNN features for the target image.
    std::vector<double> targetFeatures = computeDNNFeatures(targetImg, net);

    // Use OpenCV's glob to list all files in the database directory.
    std::vector<cv::String> imageFiles;
    std::string pattern = dbDir + "/*.*";
    cv::glob(pattern, imageFiles, false);
    if (imageFiles.empty())
    {
        std::cerr << "Error: No images found in directory " << dbDir << std::endl;
        return 1;
    }

    // Process each image in the database.
    std::vector<Match> matches;
    // Extract the basename of the target file.
    std::string targetBasename = targetImageFile.substr(targetImageFile.find_last_of("/\\") + 1);
    for (const auto &file : imageFiles)
    {
        // Skip files that are not valid images (e.g., .DS_Store, .txt, etc.).
        if (!isValidImageFile(file))
            continue;

        // Skip the target image if it is in the same directory.
        std::string basename = file.substr(file.find_last_of("/\\") + 1);
        if (basename == targetBasename)
            continue;

        cv::Mat img = cv::imread(file);
        if (img.empty())
        {
            std::cerr << "Warning: Could not load image " << file << std::endl;
            continue;
        }
        std::vector<double> features = computeDNNFeatures(img, net);
        // Always use SSD distance.
        double dist = ssd_distance(targetFeatures, features);
        matches.push_back({file, dist});
    }

    // Sort the matches in ascending order of distance.
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
              { return a.distance < b.distance; });

    // Output the top N matches.
    std::cout << "Top " << topN << " matches for target image '" << targetImageFile
              << "' using DNN features and SSD metric:\n";
    for (int i = 0; i < topN && i < matches.size(); i++)
    {
        std::cout << (i + 1) << ". " << matches[i].filename
                  << " (distance: " << matches[i].distance << ")\n";
    }

    return 0;
}