/*
    Agnibha Chatterjee
    Om Agarwal
    Feb 8 2025
    CS5330- Pattern Recognition & Computer Vision
    Reads image features from a CSV file and identifies top matching images for a target using a specified distance metric.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using FeatureVector = std::vector<double>;
using FeatureMap = std::unordered_map<std::string, FeatureVector>;

// Reads the CSV file and builds a mapping from image file name to its 512-d feature vector.
FeatureMap load_features(const std::string &csv_file)
{
    FeatureMap features;
    std::ifstream file(csv_file);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open CSV file " << csv_file << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        std::string token;
        // The first token is the filename.
        if (!std::getline(ss, token, ','))
            continue;
        std::string filename = token;
        FeatureVector fv;
        // The next tokens are the feature values.
        while (std::getline(ss, token, ','))
        {
            try
            {
                double value = std::stod(token);
                fv.push_back(value);
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Warning: Invalid numeric value for image "
                          << filename << std::endl;
            }
        }
        features[filename] = fv;
    }
    file.close();
    return features;
}

// Computes the sum-of-squares (squared Euclidean) distance between two vectors.
double ssd_distance(const FeatureVector &v1, const FeatureVector &v2)
{
    double dist = 0.0;
    if (v1.size() != v2.size())
    {
        std::cerr << "Error: Feature vector sizes do not match." << std::endl;
        return std::numeric_limits<double>::max();
    }
    for (size_t i = 0; i < v1.size(); i++)
    {
        double diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return dist;
}

// Computes the cosine distance between two vectors.
// Cosine distance = 1 - (v1 dot v2)/(||v1|| * ||v2||)
double cosine_distance(const FeatureVector &v1, const FeatureVector &v2)
{
    if (v1.size() != v2.size())
    {
        std::cerr << "Error: Feature vector sizes do not match." << std::endl;
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
        return 1.0; // define maximum distance if one vector is zero
    double cosineSim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
    return 1.0 - cosineSim;
}

// Structure for holding a match result.
struct Match
{
    std::string filename;
    double distance;
};

// Finds the top N matches for the given target image filename using the specified distance metric.
std::vector<Match> get_top_n_matches(const std::string &target_filename,
                                     const FeatureMap &features,
                                     const std::string &metric,
                                     int top_n)
{
    auto it = features.find(target_filename);
    if (it == features.end())
    {
        std::cerr << "Error: Target image " << target_filename << " not found in the features." << std::endl;
        exit(1);
    }
    const FeatureVector &target_feature = it->second;
    std::vector<Match> matches;
    for (const auto &pair : features)
    {
        if (pair.first == target_filename)
            continue; // skip the target image itself
        double dist = 0.0;
        if (metric == "ssd")
            dist = ssd_distance(target_feature, pair.second);
        else if (metric == "cosine")
            dist = cosine_distance(target_feature, pair.second);
        else
        {
            std::cerr << "Unknown distance metric: " << metric << std::endl;
            exit(1);
        }
        matches.push_back({pair.first, dist});
    }
    // Sort the matches (ascending order: smaller distance means more similar)
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
              { return a.distance < b.distance; });
    if (matches.size() > static_cast<size_t>(top_n))
        matches.resize(top_n);
    return matches;
}

// Simple helper to print usage information.
void print_usage(const std::string &prog_name)
{
    std::cout << "Usage: " << prog_name << " --target TARGET_FILENAME [--csv CSV_FILE] [--dist DISTANCE_METRIC] [--top N]\n"
              << "  --target: filename of the target image (must exist in the CSV file)\n"
              << "  --csv   : path to the CSV file containing image features (default: ResNet18_olym.csv)\n"
              << "  --dist  : distance metric to use ('cosine' or 'ssd'; default: cosine)\n"
              << "  --top   : number of top matches to return (default: 3)\n";
}

int main(int argc, char *argv[])
{
    std::string target_filename;
    std::string csv_file = "ResNet18_olym.csv";
    std::string metric = "cosine";
    int top_n = 3;

    // Basic command-line arguments processing.
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--target" && i + 1 < argc)
        {
            target_filename = argv[++i];
        }
        else if (arg == "--csv" && i + 1 < argc)
        {
            csv_file = argv[++i];
        }
        else if (arg == "--dist" && i + 1 < argc)
        {
            metric = argv[++i];
        }
        else if (arg == "--top" && i + 1 < argc)
        {
            top_n = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (target_filename.empty())
    {
        std::cerr << "Error: You must provide a target image filename using --target" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Extract the base filename if a path is provided (e.g., "../image.jpg" -> "image.jpg").
    size_t pos = target_filename.find_last_of("/\\");
    if (pos != std::string::npos)
    {
        target_filename = target_filename.substr(pos + 1);
    }

    // Load the features from CSV.
    FeatureMap features = load_features(csv_file);

    // Retrieve top matches.
    std::vector<Match> top_matches = get_top_n_matches(target_filename, features, metric, top_n);

    std::cout << "\nTop " << top_n << " matches for target image '" << target_filename
              << "' using " << metric << " distance:\n";
    for (size_t i = 0; i < top_matches.size(); i++)
    {
        std::cout << (i + 1) << ". " << top_matches[i].filename
                  << "  (distance: " << top_matches[i].distance << ")\n";
    }

    return 0;
}