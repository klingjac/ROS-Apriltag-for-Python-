# Define the compiler
CXX = g++

# Define the paths to the headers
INCLUDES = -I../ -I/usr/include/opencv4

# Define the paths to the libraries
LIBS = -L../ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_calib3d -lapriltag

# Define the source files
OPENCV_DEMO_SOURCES = opencv_demo.cc
POSE_ESTIMATION_SOURCES = pose_estimation.cpp

# Define the output executables and shared library
OPENCV_DEMO_OUTPUT = opencv_demo
POSE_ESTIMATION_OUTPUT = libpose_estimation.so

# Build the opencv_demo executable
$(OPENCV_DEMO_OUTPUT): $(OPENCV_DEMO_SOURCES)
	$(CXX) $(INCLUDES) $(OPENCV_DEMO_SOURCES) -o $(OPENCV_DEMO_OUTPUT) $(LIBS)

# Build the pose estimation shared library
$(POSE_ESTIMATION_OUTPUT): $(POSE_ESTIMATION_SOURCES)
	$(CXX) -o $(POSE_ESTIMATION_OUTPUT) -shared -fPIC $(INCLUDES) $(POSE_ESTIMATION_SOURCES) $(LIBS)

# Build both targets
all: $(OPENCV_DEMO_OUTPUT) $(POSE_ESTIMATION_OUTPUT)

# Clean up
clean:
	rm -f $(OPENCV_DEMO_OUTPUT) $(POSE_ESTIMATION_OUTPUT)
