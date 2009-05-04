#include "gpuMD5.h"

#include "cuda.h"

__global__ void md5Hash(char* message, size_t messageLength, char* digest) {
  int r[] = {7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
             5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
             4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
             6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21};

  unsigned int k[64];
  for (int i = 0; i != 64; ++i)
    k[i] = __float2uint_rz(floorf(abs(sinf(i + 1))) * powf(2, 32));
    
  unsigned int h0 = 0x67452301;
  unsigned int h1 = 0xEFCDAB89;
  unsigned int h2 = 0x98BADCFE;
  unsigned int h3 = 0x10325476;
  
  int i = 0;
  while (i < messageLength) {
    
    i += 512;
  }
}
