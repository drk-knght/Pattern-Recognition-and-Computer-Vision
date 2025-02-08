Group Members Name-
Om Agarwal
Agnibha Chatterjee

This GDrive link has the models used - https://drive.google.com/drive/folders/1iLhaYpw3POhSa1Bfk4kTQBo6jUQZzlm9?usp=sharing

Steps to run the code-
1. mkdir build
2. cd build
3. cmake ..
4. make

To run and store any feature from Q1-Q4-
./feature_extractor.cpp ./address_of_olympus_dir ./address_of_csv_file feature_type

./address_of_olympus_dir is the directory of Olympus image database
./address_of_csv_file is the csv file where you want to store the feature values for each of the implementations.
feature_type is the type of feature you want to run for. There are 4 types-
	Q1. "ssd"- sum of squared distance 
	Q2. "hist"- histogram plots
	Q3. "spatialhist"- multi-histogram matching
	Q4. "combined"- texture and color feature.

To run the image matching process against the features stored offline run the following command-
./image_matcher ./address_of_target_file ./csv_feature_file number_of_matches feature_type

./address_of_target_file is the address of the test image on which you want to run the matching algo
./csv_feature_file is the csv file which stores the feature for olympus images 
number_of_matches is the number of matches you want for the given target image when compared against the dataset.
feature_type is same as above in the feature extractor part.

Running Instructions for Q5, Q6, and Q7:
Usage:
    ./q5 --target TARGET_FILENAME [--csv CSV_FILE] [--dist DISTANCE_METRIC] [--top N]
Example:
    ./q5 --target image1.jpg --csv ResNet18_olym.csv --dist cosine --top 5

Argument Descriptions for Q5:
    --target : Filename of the target image (must exist in the CSV file).
    --csv    : Path to the CSV file containing image features. [Default: ResNet18_olym.csv]
    --dist   : Distance metric to use ('cosine' or 'ssd'). [Default: cosine]
    --top    : Number of top matches to return. [Default: 3]

Usage:
    ./q6 --target TARGET_IMAGE --db IMAGE_DIRECTORY [--top N] [--dnnmodel PATH_TO_ONNX_MODEL]
Example:
    ./q6 --target image1.jpg --db /path/to/olympus_images --top 3 --dnnmodel resnet50.onnx

Argument Descriptions for Q6:
    --target  : Path to the target image to match.
    --db      : Directory containing the image database.
    --top     : Number of top matches to return. [Default: 3]
    --dnnmodel: Path to the pre-trained ONNX model for feature extraction. [Default: resnet50.onnx]

Usage:
    ./q7 --target TARGET_IMAGE [--target2 TARGET_IMAGE2] --dir IMAGE_DIRECTORY [--dist DISTANCE_METRIC] [--top N] [--embedding-model PATH_TO_ONNX_MODEL]
Example:
    ./q7 --target image1.jpg --dir /path/to/database --top 5 --embedding-model resnet18.onnx

Argument Descriptions for Q7:
    --target         : Path to the primary target image.
    --target2        : (Optional) Path to a second target image.
    --dir            : Directory containing the database images.
    --dist           : Distance metric to use ('ssd' or 'cosine'). [Default: ssd]
    --top            : Number of top matches to return. [Default: 5]
    --embedding-model: Path to the ONNX model for DNN embedding extraction. [Default: resnet18.onnx]


Running Instructions for Extensions (matching_pipeline.cpp)
Usage:
    ./matching_pipeline <target_image> <database_directory> <method> [<top_N>]
    
Examples:
    To run matching using baseline features and display top 5 matches:
        ./matching_pipeline target.jpg /path/to/image_db baseline 5
    To run matching using the dnndsv method (all matches will be returned):
        ./matching_pipeline target.jpg /path/to/image_db dnndsv
    To run all methods for comparison:
        ./matching_pipeline target.jpg /path/to/image_db --compare-all 5

Argument Descriptions:
    <target_image>       : Path to the target image.
    <database_directory> : Directory containing the database images.
    <method>             : Image matching method to use. Options include:
                           - baseline    : Center patch feature extraction.
                           - hist        : Color histogram feature extraction.
                           - spatialhist : Spatial color histogram feature extraction.
                           - combined    : Combined texture and color feature extraction.
                           - orb         : ORB feature extraction.
                           - lbp         : Local Binary Pattern (LBP) feature extraction.
                           - ssim        : SSIM-based feature extraction.
                           - spatialvar  : Color spatial variance feature extraction.
                           - dnndsv      : Combined DNN and spatial variance feature extraction.
                                           Note: When using 'dnndsv', do not provide the <top_N> parameter.
                           - all or --compare-all : Run all methods and display results for each.
    [<top_N>]            : (Optional) The number of top matches to display.
                           Required for all methods except for 'dnndsv' (where all matches are returned).

OS- MacOS
IDE- VS Code
Time Travel Days-2

