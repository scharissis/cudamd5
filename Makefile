# Makefile:
# Generic Makefile for make-ing cuda programs
#

BIN               := cudamd5

# flags
CUDA_INSTALL_PATH := /usr/local/cuda
INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include -I$(HOME)/NVIDIA_CUDA_SDK/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(HOME)/NVIDIA_CUDA_SDK/lib
CXXFLAGS          := -O3
LDFLAGS           := -lrt -lm -lcudart -lcutil
# compilers
#NVCC              := nvcc --device-emulation
NVCC              := nvcc

# files
CPP_SOURCES       := cudamd5.cpp md5.cpp md5test.cpp deviceQuery.cpp
CU_SOURCES        := 
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) -c $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS)

cudamd5.o: cudamd5.cpp

clean:
	rm -f $(BIN) *.o *.cu_o
