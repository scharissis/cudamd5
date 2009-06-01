#define THREADS_PER_BLOCK 63
#define BLOCKS_PER_PROCESSOR 2

typedef unsigned int UINT;
typedef unsigned char UCHAR;

#include <string>
#include <vector>

std::vector<UINT> prepareTarget(const std::string&);
