CXX = g++ 
CFLAGS = -Wall -g 
CXXFLAGS = -ggdb `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`
all = facedetect

all: 
	$(CXX) $(CXXFLAGS) -o $(all) facedetect.cpp $(LDFLAGS) -lserial

.PHONY: clean

clean: 
	 rm -rf $(all) *.o
