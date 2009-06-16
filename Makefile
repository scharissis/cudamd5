BIN								:= cudamd5

# Install paths
#CUDA_TOOLKIT_PATH	:= $(HOME)/cuda/cuda
#CUDA_SDK_PATH			:= $(HOME)/cuda/sdk
#BOOST_LIB_PATH		:= 
CUDA_TOOLKIT_PATH := /usr/local/cuda
CUDA_SDK_PATH	  := /usr/local/cuda/NVIDIA_CUDA_SDK
BOOST_LIB_PATH    := /usr/local/lib

# Compilers
CXX               := g++
#NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc -deviceemu
NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc

# Compiler flags
INCLUDES          := -I. -I$(CUDA_TOOLKIT_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_PATH)/lib -L$(BOOST_LIB_PATH)
LDFLAGS           := -lcudart -lboost_program_options
CXXFLAGS  				:= -O3 -Wall -Werror
NVCCFLAGS         := -O3

# Source files
CPP_SOURCES       := CudaMD5.cpp
CU_SOURCES        := MD5_GPU.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(LIBS) $(LDFLAGS) -o $(BIN) $(CU_OBJS) $(CPP_OBJS)

clean:
	rm -f $(BIN) *.o
