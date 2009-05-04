# Makefile:
# Generic Makefile for make-ing cuda programs
#

BIN               := cudamd5

HAX		  := /usr/local/lib/libboost_program_options-gcc42-mt-1_38.a

# flags
CUDA_INSTALL_PATH := /usr/local/cuda
SDK_PATH	  := /home/stefano/NVIDIA_CUDA_SDK
BOOST_LIB_PATH    := /usr/local/lib

# U S E  W I T H  C U D A
INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_PATH)/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(HOME)/$(SDK_PATH)/lib -L$(BOOST_LIB_PATH)
LDFLAGS           := -lrt -lm -lcudart
CXXFLAGS				:= -O3

# U S E  W I T H O U T  C U D A
#LDFLAGS				:= -lboost_program_options

# compilers
#NVCC              := nvcc --device-emulation
NVCC              := nvcc

# files
CPP_SOURCES       := cudamd5.cpp md5.cpp md5test.cpp Permutator.cpp deviceQuery.cpp
CU_SOURCES        := 
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) -c $(INCLUDES) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(HAX)

cudamd5.o: cudamd5.cpp

clean:
	rm -f $(BIN) *.o *.cu_o
