BIN								:= cudamd5

# Install paths
CUDA_TOOLKIT_PATH	:= $(HOME)/cuda/cuda
CUDA_SDK_PATH			:= $(HOME)/cuda/sdk
BOOST_LIB_PATH		:= /import/kamen/1/se3010/soft/install-boost/lib

# Compilers
CXX               := g++
NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc --device-emulation
#NVCC              := $(CUDA_TOOLKIT_PATH)/bin/nvcc

# Compiler flags
INCLUDES          := -I. -I$(CUDA_TOOLKIT_PATH)/include -I$(CUDA_SDK_PATH)/common/inc -I/import/kamen/1/se3010/soft/install-boost_1_39_0/include/boost-1_39
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_PATH)/lib -L$(BOOST_LIB_PATH)
LDFLAGS           := -lcudart -lboost_program_options-gcc42-mt
CXXFLAGS  				:= -O3 -Wall -Werror
NVCCFLAGS         := -O3

# Source files
CPP_SOURCES       := Permutator.cpp Utility.cpp cudamd5.cpp
CU_SOURCES        := md5GPU.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(BIN) *.o
