#include "md5GPU.h"

#include <string>
#include <vector>

#include <cuda.h>

__constant__ uint4 dTarget;

using namespace std;

void initialiseGPU(const string& target) {
  // Create the target on the host.
  vector<UINT> t = prepareTarget(target);
  uint4 hTarget(t[0], t[1], t[2], t[3]);
  
  cudaMemcpyToSymbol(dTarget, hTarget, sizeof(hTarget));
}

std::string findTarget(const vector<string>& messages) {

}