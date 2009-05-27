#include "gpuMD5.h"

#include "cuda.h"

extern __shared__ UINT array[];
//__shared__ UINT paddedMessage[];
//__shared__ UCHAR m[];

__global__ void md5Hash(char*, UINT*, int);
__device__ UINT* pad(UINT*, UCHAR*, int);

void doHash(char* a, UINT* d, UINT* h, int length) {
  cudaMalloc((void**)&d, 64);
  md5Hash <<< 1, 2, 8000 >>> (a, d, length);
  cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost);
}

__global__ void md5Hash(char* message, UINT* digest, int length) {

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
  /*
  UCHAR message[4];
  
  //cudaMalloc((void**)&message, strlen(message));
  
  message[0] = 'a';
  message[1] = 'a';
  message[2] = 'a';
  message[3] = 'a';
  */
  pad(digest, (UCHAR*)message, length);
}

__device__ UINT* pad(UINT* paddedMessage, UCHAR* message, int msgLength) {
  //UCHAR* m;
  //UINT* paddedMessage = 0;
  //UINT* paddedMessage = (UINT*)array;
  //UCHAR* m = (UCHAR*)&paddedMessage[16];
//UINT* paddedMessage = (UINT*)&array;
UCHAR m[56];
  
  //cudaMalloc((void**)&m, 56 * sizeof(UCHAR));
  
  int i;
  for (i = 0; i != 56; ++i)
    m[i] = 0x00;

  for (i = 0; i != msgLength; ++i)
    m[i] = message[i];

  //cudaMemset(m, 0x00, 56);
  //cudaMemcpy(m, message, msgLength);
  m[msgLength] = 0x80;
  
  //cudaMalloc((void**)&paddedMessage, 16 * sizeof(UINT));
  
  //int i = 0;
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
