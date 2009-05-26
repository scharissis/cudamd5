#include "gpuMD5.h"

#include "cuda.h"

__global__ void md5Hash(UINT* digest) {
  /*
  int r[] = {7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
             5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
             4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
             6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21};

  // This can be optimized if we just hardcode the values directly.
  unsigned int k[64];
  for (int i = 0; i != 64; ++i)
    k[i] = __float2uint_rz(floorf(abs(sinf(i + 1))) * powf(2, 32));
    
  unsigned int h0 = 0x67452301;
  unsigned int h1 = 0xEFCDAB89;
  unsigned int h2 = 0x98BADCFE;
  unsigned int h3 = 0x10325476;
  
  int i = 0;
  while (i < 16) {//
    
    i += 512;
  }
  */
  UCHAR* message;
  
  cudaMalloc((void**)&message, 4);
  
  message[0] = 'a';
  message[1] = 'a';
  message[2] = 'a';
  message[3] = 'a';
  
  digest = pad(message, 4);
}

__device__ UINT* pad(UCHAR* message, int msgLength) {
  UCHAR* m;
  UINT* paddedMessage = 0;
  
  cudaMalloc((void**)&m, 56 * sizeof(UCHAR));
  
  memset(m, 0x00, 56);
  memcpy(m, message, msgLength);
  m[msgLength] = 0x80;
  
  cudaMalloc((void**)&paddedMessage, 16 * sizeof(UINT));
  
  int i = 0;
  for (i = 0; i != 14; ++i) {
      paddedMessage[i] = (UINT)m[i*4+3] << 24 |
        (UINT)m[i*4+2] << 16 |
        (UINT)m[i*4+1] << 8 |
        (UINT)m[i*4];
  }
  
  paddedMessage[14] = msgLength << 3;
  paddedMessage[15] = msgLength >> 29;
  
  return paddedMessage;
}

__device__ UINT* transform(UINT* chunk) {
  return 0;
}

/*
__device__ void hash(UINT* message, UINT* digest) {

}
abcd efgh

dcba hgfe
*/
