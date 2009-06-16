#include "gpuMD5.h"
#include "cutil.h"

// Forward declarations of basic MD5 functions
__device__ UINT F(UINT x, UINT y, UINT z);
__device__ UINT G(UINT x, UINT y, UINT z);
__device__ UINT H(UINT& x, UINT& y, UINT& z);
__device__ UINT I(UINT& x, UINT& y, UINT& z);

__device__ UINT ROTATE_LEFT(UINT& x, UINT& n);

__device__ void FF(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void GG(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void HH(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void II(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);

// F, G and H are basic MD5 functions: selection, majority, parity 
__device__ UINT F(UINT x, UINT y, UINT z){
  return ((x & y) | (~x & z));
}
__device__ UINT G(UINT x, UINT y, UINT z){
  return ((x & z) | (y & ~z));
}
__device__ UINT H(UINT& x, UINT& y, UINT& z){
  return (x ^ y ^ z);
}
__device__ UINT I(UINT& x, UINT& y, UINT& z){
  return (y ^ (x | ~z));
}

// ROTATE_LEFT rotates x left n bits 
__device__ UINT ROTATE_LEFT(UINT& x, UINT& n){
  return ((x << n) | (x >> (32 - n)));
}

// FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4
// Rotation is separate from addition to prevent recomputation __device__ void FF(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){
  a += F(b, c, d) + x + ac; 
  a =  ROTATE_LEFT(a, s); 
  a += b;
}
__device__ void GG(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){
  a += G(b, c, d) + x + ac; 
  a =  ROTATE_LEFT(a, s); 
  a += b;
}
__device__ void HH(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){  
  (a) += H ((b), (c), (d)) + (x) + (UINT)(ac); 
   (a) = ROTATE_LEFT ((a), (s)); 
   (a) += (b); 
}
__device__ void II(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){ 
(a) += I ((b), (c), (d)) + (x) + (UINT)(ac); 
   (a) = ROTATE_LEFT ((a), (s)); 
   (a) += (b);
}
// CONSTANT DECLARATIONS
extern __shared__ char array[];
//__constant__ int device_shift_amounts[64];
//__constant__ UINT device_sines[64];
__constant__ UINT deviceTarget[4];
__device__ int resultIndex;

__global__ void md5Hash(UCHAR**, int*, uint4*);
__device__ UINT* pad(UCHAR*, int);

void initialiseConstants(UINT* target) {
  UINT nf = -1;
/*
  int host_shift_amounts[] = {7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
             5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
             4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
             6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21};
  cudaMemcpyToSymbol(device_shift_amounts, host_shift_amounts, sizeof(host_shift_amounts));

  UINT host_sines[] = {3614090360, 3905402710, 606105819, 3250441966, 4118548399, 1200080426, 2821735955,
                       4249261313, 1770035416, 2336552879, 4294925233, 2304563134, 1804603682, 4254626195, 
                       2792965006, 1236535329, 4129170786, 3225465664, 643717713, 3921069994, 3593408605,
                       38016083, 3634488961, 3889429448, 568446438, 3275163606, 4107603335, 1163531501,
                       2850285829, 4243563512, 1735328473, 2368359562, 4294588738, 2272392833, 1839030562,
                       4259657740, 2763975236, 1272893353, 4139469664, 3200236656, 681279174, 3936430074,
                       3572445317, 76029189, 3654602809, 3873151461, 530742520, 3299628645, 4096336452,
                       1126891415, 2878612391, 4237533241, 1700485571, 2399980690, 4293915773, 2240044497,
                       1873313359, 4264355552, 2734768916, 1309151649, 4149444226, 3174756917, 718787259, 3951481745};                       
  
  cudaMemcpyToSymbol(device_sines, host_sines, sizeof(host_sines));
*/
  // Copy target hash to the device (So that the comparison can be done on the GPU)
  cudaMemcpyToSymbol(deviceTarget, target, sizeof(deviceTarget));
  cudaMemcpyToSymbol(resultIndex, &nf, sizeof(nf));
  
}


// md5Hash(message, device, host, length);
bool doHash(std::vector<std::string>& keys) {
  using namespace std;

  // Getting the device properties.
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int numBlocks = deviceProp.multiProcessorCount * 2;
  int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
  int numThreadsPerGrid = numBlocks * numThreadsPerBlock;
  int sharedMem = deviceProp.sharedMemPerBlock / 2;

  // Array of pointers to each message on device memory.
  UCHAR* hostMsgLocationsOnDevice[numThreadsPerGrid];
  UCHAR** deviceMsgLocationsOnDevice;
  
  // Array of lengths of each message.
  int hostMsgLengths[numThreadsPerGrid];
  int* deviceMsgLengths;

  // Array of hash for each message.
  uint4 hostDigests[numThreadsPerGrid];
  uint4* deviceDigests;
  UCHAR result[56];

  cudaError_t err = cudaGetLastError();
  
// For each message
  for (int i = 0; i != keys.size(); ++i) {
    const char* key = keys[i].c_str();
    hostMsgLengths[i] = keys[i].size();
    
    // Create an array of pointers pointing to the hashes
    cudaMalloc((void **)&hostMsgLocationsOnDevice[i], keys[i].length() * sizeof(UCHAR));
    cudaMemcpy(hostMsgLocationsOnDevice[i], key, keys[i].length(), cudaMemcpyHostToDevice);
    
  err = cudaGetLastError();
  if (cudaSuccess != err)
    printf("1: %s\n", cudaGetErrorString(err));
    cudaMalloc((void **)&deviceMsgLengths, numThreadsPerGrid * sizeof(int));
    cudaMemcpy(deviceMsgLengths, hostMsgLengths, numThreadsPerGrid * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaSuccess != err)
    printf("2: %s\n", cudaGetErrorString(err));
  }
  
  cudaMalloc((void **)&deviceMsgLocationsOnDevice, numThreadsPerGrid * sizeof(UCHAR*));
  cudaMemcpy(deviceMsgLocationsOnDevice, hostMsgLocationsOnDevice, sizeof(hostMsgLocationsOnDevice), cudaMemcpyHostToDevice);
  if (cudaSuccess != err)
    printf("3: %s\n", cudaGetErrorString(err));

  
  cudaMalloc((void **)&deviceDigests, numThreadsPerGrid * sizeof(uint4));
  err = cudaGetLastError();
  if (cudaSuccess != err)
    printf("4: %s\n", cudaGetErrorString(err));
    
  md5Hash <<< numBlocks, numThreadsPerBlock, sharedMem >>> (deviceMsgLocationsOnDevice, deviceMsgLengths, deviceDigests);
  //cudaThreadSynchronize();
  
    err = cudaGetLastError();
  if (cudaSuccess != err)
    printf("5: %s\n", cudaGetErrorString(err));
  cudaMemcpy(hostDigests, deviceDigests, sizeof(hostDigests), cudaMemcpyDeviceToHost);
  int ri = NOT_FOUND;
  int* resultAddress;
  //  cudaGetSymbolAddress((void**)&resultAddress, "resultIndex");
  
  
  if (cudaSuccess == cudaGetSymbolAddress((void**)&resultAddress, "resultIndex")) {
    //printf("OK\n");
  } else {
    printf("ERROR: Address of resultAddress is invalid!\n");
  };
  
  
  cudaMemcpy(&ri, resultAddress, sizeof(int), cudaMemcpyDeviceToHost);
  // Check if target hash was found
  if (ri >= 0){
    //printf("Result index: %d\n",ri);
    cudaMemcpy(result, hostMsgLocationsOnDevice[ri], hostMsgLengths[ri], cudaMemcpyDeviceToHost);
    printf("Hash found: ");
    //Prints 
    for(int i =0; i != hostMsgLengths[ri]; i++)
      putchar(result[i]);
    putchar('\n');
    
    return true;
    
    //printf("Data at result %08x %08x %08x %08x\n",result[0],result[1],result[2],result[3]);
  } else {
    return false;
    //printf("Target hash not found.\n");
  }
  err = cudaGetLastError();
  if (cudaSuccess != err)
    printf("9: %s\n", cudaGetErrorString(err));

  // Free memory
  cudaFree(hostMsgLocationsOnDevice);
  cudaFree(deviceMsgLengths);  
}

__global__ void md5Hash(UCHAR** messages, int* msgLengths, uint4* digests) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  UINT* in = pad(messages[idx], msgLengths[idx]);
  
  //digests[idx] = make_uint4(paddedMessage[0], paddedMessage[1], paddedMessage[2], paddedMessage[14]);

  unsigned int h0 = 0x67452301;
  unsigned int h1 = 0xEFCDAB89;
  unsigned int h2 = 0x98BADCFE;
  unsigned int h3 = 0x10325476;
  
  UINT a = h0;
  UINT b = h1;
  UINT c = h2;
  UINT d = h3;
  
    /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF ( a, b, c, d, in[ 0], S11, 3614090360); /* 1 */
  FF ( d, a, b, c, in[ 1], S12, 3905402710); /* 2 */
  FF ( c, d, a, b, in[ 2], S13,  606105819); /* 3 */
  FF ( b, c, d, a, in[ 3], S14, 3250441966); /* 4 */
  FF ( a, b, c, d, in[ 4], S11, 4118548399); /* 5 */
  FF ( d, a, b, c, in[ 5], S12, 1200080426); /* 6 */
  FF ( c, d, a, b, in[ 6], S13, 2821735955); /* 7 */
  FF ( b, c, d, a, in[ 7], S14, 4249261313); /* 8 */
  FF ( a, b, c, d, in[ 8], S11, 1770035416); /* 9 */
  FF ( d, a, b, c, in[ 9], S12, 2336552879); /* 10 */
  FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */
  FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */
  FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */
  FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */
  FF ( c, d, a, b, in[14], S13, 2792965006); /* 15 */
  FF ( b, c, d, a, in[15], S14, 1236535329); /* 16 */
  
  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, in[ 1], S21, 4129170786); /* 17 */
  GG ( d, a, b, c, in[ 6], S22, 3225465664); /* 18 */
  GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */
  GG ( b, c, d, a, in[ 0], S24, 3921069994); /* 20 */
  GG ( a, b, c, d, in[ 5], S21, 3593408605); /* 21 */
  GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */
  GG ( c, d, a, b, in[15], S23, 3634488961); /* 23 */
  GG ( b, c, d, a, in[ 4], S24, 3889429448); /* 24 */
  GG ( a, b, c, d, in[ 9], S21,  568446438); /* 25 */
  GG ( d, a, b, c, in[14], S22, 3275163606); /* 26 */
  GG ( c, d, a, b, in[ 3], S23, 4107603335); /* 27 */
  GG ( b, c, d, a, in[ 8], S24, 1163531501); /* 28 */
  GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */
  GG ( d, a, b, c, in[ 2], S22, 4243563512); /* 30 */
  GG ( c, d, a, b, in[ 7], S23, 1735328473); /* 31 */
  GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, in[ 5], S31, 4294588738); /* 33 */
  HH ( d, a, b, c, in[ 8], S32, 2272392833); /* 34 */
  HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */
  HH ( b, c, d, a, in[14], S34, 4259657740); /* 36 */
  HH ( a, b, c, d, in[ 1], S31, 2763975236); /* 37 */
  HH ( d, a, b, c, in[ 4], S32, 1272893353); /* 38 */
  HH ( c, d, a, b, in[ 7], S33, 4139469664); /* 39 */
  HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */
  HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */
  HH ( d, a, b, c, in[ 0], S32, 3936430074); /* 42 */
  HH ( c, d, a, b, in[ 3], S33, 3572445317); /* 43 */
  HH ( b, c, d, a, in[ 6], S34,   76029189); /* 44 */
  HH ( a, b, c, d, in[ 9], S31, 3654602809); /* 45 */
  HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */
  HH ( c, d, a, b, in[15], S33,  530742520); /* 47 */
  HH ( b, c, d, a, in[ 2], S34, 3299628645); /* 48 */
  
    /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, in[ 0], S41, 4096336452); /* 49 */
  II ( d, a, b, c, in[ 7], S42, 1126891415); /* 50 */
  II ( c, d, a, b, in[14], S43, 2878612391); /* 51 */
  II ( b, c, d, a, in[ 5], S44, 4237533241); /* 52 */
  II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */
  II ( d, a, b, c, in[ 3], S42, 2399980690); /* 54 */
  II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */
  II ( b, c, d, a, in[ 1], S44, 2240044497); /* 56 */
  II ( a, b, c, d, in[ 8], S41, 1873313359); /* 57 */
  II ( d, a, b, c, in[15], S42, 4264355552); /* 58 */
  II ( c, d, a, b, in[ 6], S43, 2734768916); /* 59 */
  II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */
  II ( a, b, c, d, in[ 4], S41, 4149444226); /* 61 */
  II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */
  II ( c, d, a, b, in[ 2], S43,  718787259); /* 63 */
  II ( b, c, d, a, in[ 9], S44, 3951481745); /* 64 */
  
  a += h0;
  b += h1;
  c += h2;
  d += h3;
  
  // Check to see if this is the target hash!
  if (a == deviceTarget[0] && b == deviceTarget[1] && c == deviceTarget[2] && d == deviceTarget[3]){
     resultIndex = idx;
  } else {
    //resultIndex = -1;  // For some reason this does not work!? (Program wont find target hash)
  }
  digests[idx] = make_uint4(a, b, c, d);
  
}

__device__ UINT* pad(UCHAR* message, int msgLength) {
  //UCHAR* m;
  //UINT* paddedMessage = 0;
  //UINT* paddedMessage = (UINT*)array;
  //UCHAR* m = (UCHAR*)&paddedMessage[16];
  //UINT* paddedMessage = (UINT*)&array;
//  UCHAR* m = (UCHAR*)&array + (blockIdx.x * threadIdx.x) ;
  
  UCHAR m[56];
  
  for (int i = 0; i != 56; ++i)
    m[i] = 0x00;
  
  
  for (int i = 0; i != msgLength; ++i)
    m[i] = message[i];
  
  m[msgLength] = 0x80;
  
  UINT* paddedMessage = ((UINT*)&array) + (((blockIdx.x * blockDim.x + threadIdx.x) * 16));
  
  for (int i = 0; i != 14; ++i) {
      paddedMessage[i] = (UINT)m[i*4+3] << 24 |
        (UINT)m[i*4+2] << 16 |
        (UINT)m[i*4+1] << 8 |
        (UINT)m[i*4];
  }
  
  paddedMessage[14] = msgLength << 3;
  paddedMessage[15] = msgLength >> 29;

  return paddedMessage;
}

