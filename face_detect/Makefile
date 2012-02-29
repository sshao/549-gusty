CXX = g++ -lserial
CFLAGS = -Wall -g 
CXXFLAGS = `pkg-config opencv --cflags`
LDFLAGS = `pkg-config opencv --libs`
all = facedetect 

all: $(all)

.PHONY: clean

clean: 
	 rm -rf $(all) *.o
