# MacOS compiler
CC = g++

# Make it the C++ compiler too
CXX = $(CC)

# OSX include paths (for Homebrew)
CFLAGS = -Wc++11-extensions -std=c++14 -I/opt/homebrew/Cellar/opencv/4.11.0/include/opencv4 -I/opt/homebrew/include -I/opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime -DENABLE_PRECOMPILED_HEADERS=OFF -Wc++14-extensions

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to where your libraries are stored
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.11.0/lib -L/opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib -L/opt/homebrew/lib

# opencv libraries and MacOS frameworks
# these settings work for macOS and macports
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime

# Location to write the executables
BINDIR = .

# rules for making various executables
mnist: mnist-cv.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

depth: depthanything.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/local/lib ./$@

da2: da2-example.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/local/lib ./$@

da2vid: da2-video.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/local/lib ./$@

clean:
	rm -f *.o *~ 
