#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>

int main(int argc, char *argv[]){
    cv::Mat src;
    cv::Mat dst;
    char filename[256];

    if(argc<2){
        printf("usage: %s <image filename>\n",argv[0]);
        exit(-1);
    }
    strncpy(filename,argv[1],255);

    src=cv::imread(filename);
    if(src.data==NULL){
        printf("error: unable to read image %s\n",filename);
        exit(-2);
    }
    cv::imshow(filename,src);
    while(1){
        int c=cv::waitKey(0); // blocking call with an argument of 0
        if(c==(int)('q')){
            break;
        }
    }
    
    printf("Terminating\n");
  
    return(0);
}

