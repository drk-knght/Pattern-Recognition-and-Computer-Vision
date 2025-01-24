#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
    cv::Mat src;        // Source image matrix
    cv::Mat dst;        // Destination image matrix (not used in this code)
    char filename[256]; // Buffer to hold the image filename

    // Check if the user provided an image filename as an argument
    if (argc < 2)
    {
        printf("usage: %s <image filename>\n", argv[0]); // Print usage message
        exit(-1);                                        // Exit if no filename is provided
    }
    strncpy(filename, argv[1], 255); // Copy the filename from command line argument

    // Read the image from the specified filename
    src = cv::imread(filename);
    // Check if the image was successfully loaded
    if (src.data == NULL)
    {
        printf("error: unable to read image %s\n", filename); // Print error message
        exit(-2);                                             // Exit if the image cannot be read
    }
    cv::imshow(filename, src); // Display the image in a window
    while (1)
    {
        int c = cv::waitKey(0); // Wait for a key press indefinitely
        // Check if the 'q' key was pressed to exit the loop
        if (c == (int)('q'))
        {
            break; // Break the loop if 'q' is pressed
        }
    }

    printf("Terminating\n"); // Print termination message

    return (0); // Return success
}
