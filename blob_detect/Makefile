CXX = g++ -lserial
CFLAGS = -Wall -g 
CXXFLAGS = `pkg-config opencv cvblob --cflags`
LDFLAGS = `pkg-config opencv cvblob --libs`
all = blobdetect 

all: $(all)

.PHONY: clean

clean: 
	 rm -rf $(all) *.o
