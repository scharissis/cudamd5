#include "gpuMD5.h"

extern __shared__ char array[];

__global__ void md5Hash(UCHAR**, int*, uint4*);
__device__ UINT* pad(UINT*, UCHAR*, int);

// md5Hash(message, device, host, length);
void doHash(std::vector<std::string>& keys) {
  using namespace std;

  // Getting the device properties.
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numBlocks = deviceProp.multiProcessorCount * 2;
  int numThreadsPerGrid = 64 * numBlocks;
  int sharedMem = deviceProp.sharedMemPerBlock / 2;

  // Array of pointers to each message on device memory.
  UCHAR* msgLocationsOnDevice[numThreadsPerGrid];
  
  // Array of lengths of each message.
  int hostMsgLengths[numThreadsPerGrid];
  int* deviceMsgLengths;

  // Array of hash for each message.
  uint4 hostDigests[numThreadsPerGrid];
  uint4 deviceDigests[numThreadsPerGrid];

  // For each message
  for (int i = 0; i != keys.size(); ++i) {
    const char* key = keys[i].c_str();
    
    cudaMalloc((void **)&msgLocationsOnDevice[i], keys[i].length() * sizeof(UCHAR));
    cudaMemcpy(msgLocationsOnDevice[i], key, keys[i].length(), cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&deviceMsgLengths, numThreadsPerGrid * sizeof(int));
    cudaMemcpy(deviceMsgLengths, hostMsgLengths, numThreadsPerGrid * sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaMalloc((void **)&deviceDigests, numThreadsPerGrid * sizeof(uint4));
  
  md5Hash <<< numThreadsPerGrid, numBlocks, sharedMem >>> (msgLocationsOnDevice, deviceMsgLengths, deviceDigests);


/*
    char* b;
    UINT h[16];
    UINT* d;

    cudaMalloc((void **)&b, t.size() * sizeof(char));
    cudaMemcpy(b, a, t.size(), cudaMemcpyHostToDevice); 

    //doHash(b, d, h, t.size());
    
    cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost);
    
      int i;
  for (i = 0; i != 16; ++i)
    printf("%2d %08x\n", i, h[i]);
    
    cudaFree(d);

  cudaMalloc((void**)&d, 64);
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int numBlocks = deviceProp.multiProcessorCount * 2;
  int numThreadsPerGrid = 64 * numBlocks;
  int sharedMem = deviceProp.sharedMemPerBlock / 2;

  md5Hash <<< numThreadsPerGrid, numBlocks, sharedMem >>> (a, d, length);
  
  cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost);
*/
}

__global__ void md5Hash(UCHAR** messages, int* msgLengths, uint4* digests) {

  //pad(digest, (UCHAR*)message, numMessages);

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
    
  
}

__device__ UINT* pad(UINT* paddedMessage, UCHAR* message, int msgLength) {
  //UCHAR* m;
  //UINT* paddedMessage = 0;
  //UINT* paddedMessage = (UINT*)array;
  //UCHAR* m = (UCHAR*)&paddedMessage[16];
  //UINT* paddedMessage = (UINT*)&array;
  UCHAR* m = (UCHAR*)&array;;
  
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
