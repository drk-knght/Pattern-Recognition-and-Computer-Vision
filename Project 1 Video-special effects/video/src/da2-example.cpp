/*
  Depth-aware Face Detection

  Combines DA2Network depth detection with face detection to create
  depth-aware visual effects around detected faces.
*/

#include <cstdio>
#include <cstring>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "faceDetect.h"

int main(int argc, char *argv[])
{
  cv::VideoCapture *capdev;
  cv::Mat src;
  cv::Mat dst;
  cv::Mat dst_vis;
  cv::Mat grey;
  std::vector<cv::Rect> faces;
  const float reduction = 0.5;

  // make a DANetwork object
  DA2Network da_net("model_fp16.onnx");

  // open the video device
  capdev = new cv::VideoCapture(0);
  if (!capdev->isOpened())
  {
    printf("Unable to open video device\n");
    return (-1);
  }

  cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 256.0 / (refS.height * reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  cv::namedWindow("Video", 1);
  cv::namedWindow("Depth", 1);
  cv::namedWindow("Depth-Aware Faces", 1);

  for (;;)
  {
    *capdev >> src;
    if (src.empty())
    {
      printf("frame is empty\n");
      break;
    }

    // Reduce frame size for speed
    cv::resize(src, src, cv::Size(), reduction, reduction);

    // Get depth information
    da_net.set_input(src, scale_factor);
    da_net.run_network(dst, src.size());
    cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO);

    // Detect faces
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    detectFaces(grey, faces);

    // Create a copy for depth-aware effects
    cv::Mat effect_frame = src.clone();

    // For each detected face, apply depth-aware effects
    for (const auto &face : faces)
    {
      // Calculate average depth in face region
      cv::Mat face_depth = dst(face);
      cv::Scalar avg_depth = cv::mean(face_depth);

      // Create different effects based on depth
      if (avg_depth[0] < 85)
      { // Close face
        // Red glow effect for close faces
        cv::rectangle(effect_frame, face, cv::Scalar(0, 0, 255), 3);
        cv::putText(effect_frame, "CLOSE",
                    cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 2);
      }
      else if (avg_depth[0] < 170)
      { // Medium distance
        // Yellow box for medium distance
        cv::rectangle(effect_frame, face, cv::Scalar(0, 255, 255), 2);
        cv::putText(effect_frame, "MEDIUM",
                    cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 2);
      }
      else
      { // Far face
        // Green thin box for far faces
        cv::rectangle(effect_frame, face, cv::Scalar(0, 255, 0), 1);
        cv::putText(effect_frame, "FAR",
                    cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
      }
    }

    // Display results
    cv::imshow("Video", src);
    cv::imshow("Depth", dst_vis);
    cv::imshow("Depth-Aware Faces", effect_frame);

    char key = cv::waitKey(10);
    if (key == 'q')
    {
      break;
    }
    // Save images with 's'
    else if (key == 's')
    {
      cv::imwrite("depth_image.png", dst_vis);
      cv::imwrite("depth_aware_faces.png", effect_frame);
      printf("Images saved\n");
    }
  }

  printf("Terminating\n");
  delete capdev;
  return (0);
}
