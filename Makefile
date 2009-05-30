BIN               := cudamd5

HAX := /usr/local/lib/libboost_program_options-gcc42-mt-1_38.a

# Install paths
#CUDA_TOOLKIT_PATH := $(HOME)/cuda/cuda
#CUDA_SDK_PATH     := $(HOME)/cuda/sdk
#BOOST_LIB_PATH    := /usr/lib
CUDA_TOOLKIT_PATH := /usr/local/cuda
CUDA_SDK_PATH	  := /usr/local/cuda/NVIDIA_CUDA_SDK
BOOST_LIB_PATH    := /usr/local/lib

# Compilers
CXX               := g++
#NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc --device-emulation
NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc

# Compiler flags
INCLUDES          := -I. -I$(CUDA_TOOLKIT_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_PATH)/lib -L$(BOOST_LIB_PATH)
#LDFLAGS           := -lcudart -lboost_program_options
LDFLAGS := -lcudart
CXXFLAGS  				:= -O3 -Wall
NVCCFLAGS         := -O3

# Source files
CPP_SOURCES       := md5.cpp md5test.cpp Permutator.cpp cudamd5.cpp
CU_SOURCES        := deviceQuery.cu gpuMD5.cu 
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS) $(LDFLAGS) $(HAX)

clean:
	rm -f $(BIN) *.o
