BIN               := cudamd5

# Install paths
CUDA_TOOLKIT_PATH := $(HOME)/cuda/cuda
CUDA_SDK_PATH     := $(HOME)/cuda/sdk
BOOST_LIB_PATH    := /usr/lib

# Compilers
CXX               := g++
NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc --device-emulation
#NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc

# Compiler flags
INCLUDES          := -I. -I$(CUDA_TOOLKIT_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_PATH)/lib -L$(BOOST_LIB_PATH)
LDFLAGS           := -lcudart -lboost_program_options
CXXFLAGS  				:= -O3 -Wall
NVCCFLAGS         := -O3

# Source files
CPP_SOURCES       := md5.cpp md5test.cpp Permutator.cpp
CU_SOURCES        := deviceQuery.cu arrayTest.cu gpuMD5.cu cudamd5.cu 
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(BIN) *.o
