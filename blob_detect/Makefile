CXX = g++ 
CFLAGS = -Wall -g 
CXXFLAGS = -ggdb `pkg-config --cflags opencv cvblob`
LDFLAGS = `pkg-config --libs opencv cvblob`
all = blobdetect

all: 
	$(CXX) $(CXXFLAGS) -o $(all) blobdetect.cpp $(LDFLAGS) -lserial

.PHONY: clean

clean: 
	 rm -rf $(all) *.o
