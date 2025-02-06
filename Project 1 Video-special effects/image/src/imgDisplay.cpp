#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
    // Check for image filename argument
    if (argc < 2)
    {
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }

    char filename[256];
    strncpy(filename, argv[1], 255);
    filename[255] = '\0'; // Ensure null termination

    // Load the input image
    cv::Mat src = cv::imread(filename);
    if (src.empty())
    {
        printf("error: unable to read image %s\n", filename);
        exit(-2);
    }

    // Convert the image to grayscale since gradient operations are typically performed on a single channel
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Compute gradients along x and y directions using the Sobel operator
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3); // x-direction gradient
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3); // y-direction gradient

    // Compute the gradient magnitude for each pixel: sqrt(grad_x^2 + grad_y^2)
    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    // Compute the average energy of the gradient magnitude image
    cv::Scalar avg_energy = cv::mean(grad_mag);
    printf("Average gradient energy: %f\n", avg_energy[0]);
    printf("Observation: The average energy of gradient magnitude can be a useful feature for differentiating textures.\n");
    printf("Rough textures will tend to have a higher gradient energy, whereas smoother textures result in lower gradient energy.\n");

    // Convert gradient magnitude to an 8-bit image for display purposes
    cv::Mat grad_display;
    cv::convertScaleAbs(grad_mag, grad_display);

    // Display the original image and the gradient magnitude image
    cv::imshow("Original Image", src);
    cv::imshow("Gradient Magnitude", grad_display);

    printf("Press 'q' to quit the display windows.\n");
    // Wait until the user presses 'q' or 'Q'
    while (true)
    {
        int c = cv::waitKey(0);
        if (c == 'q' || c == 'Q')
        {
            break;
        }
    }

    printf("Terminating\n");
    return 0;
}
