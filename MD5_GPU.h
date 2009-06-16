// MD5_GPU.h

#include <string>
#include <utility>

#include "Utility.h"

void initialiseGPU(std::string, std::string);

std::pair<bool, std::string> findMessage(size_t, size_t, size_t);

